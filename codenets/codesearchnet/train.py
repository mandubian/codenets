#!/usr/bin/env python3
"""
Usage:
    multi_branch_train.py [options]
    multi_branch_train.py [options]

Options:
    -h --help                        Show this screen.
    --config FILE                    Specify HOCON config file.
    --restore DIR                    specify restoration dir. [optional]
    --debug                          Enable debug routines. [default: False]
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from docopt import docopt
from dpu_utils.utils import run_and_debug
from loguru import logger
from pyhocon import ConfigFactory, ConfigTree
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import wandb

from codenets.codesearchnet.dataset_utils import BalancedBatchSchedulerSampler, DatasetType
from codenets.save import save_records_best, save_records_last
from codenets.codesearchnet.training_ctx import (
    CodeSearchTrainingContext,
    compute_loss_mrr,
    AvgLoss,
    AvgMrr,
    UsedTime,
    TotalLoss,
    TotalMrr,
    TotalSize,
    BatchSize,
    BatchLoss,
)

logger.remove()
logger.add(sys.stderr, level="DEBUG", colorize=True, backtrace=False)


@dataclass
class EpochResult:
    loss: AvgLoss
    mrr: AvgMrr
    used_time: UsedTime


def run_epoch(
    prefix: str,
    epoch: int,
    training_ctx: CodeSearchTrainingContext,
    dataloader: torch.utils.data.DataLoader,
    log_interval: int,
    is_train: bool = True,
) -> EpochResult:
    """
    Run epoch computation according to is_train flag
    Please note that IT MUTATES training_ctx
    (We need to modify models & optimizer in all cases so let's mutate)
    """
    total_loss = TotalLoss(0.0)
    total_size = TotalSize(0)
    total_mrr = TotalMrr(0.0)
    training_ctx.epoch = epoch
    epoch_start = time.time()
    if is_train:
        logger.debug("Train Mode")
        training_ctx.train_mode()
    else:
        logger.debug("Eval Mode")
        training_ctx.eval_mode()

    training_ctx.zero_grad()
    with tqdm(total=len(dataloader)) as t_batch:
        for batch_idx, batch in enumerate(dataloader):
            per_sample_loss, per_sample_losses, similarity_scores = training_ctx.forward(batch, batch_idx)

            if is_train:
                training_ctx.backward_optimize(per_sample_loss)
                training_ctx.zero_grad()

            batch_size = BatchSize(batch[0].size()[0])
            batch_loss = BatchLoss(per_sample_loss.item())
            total_loss, avg_loss, total_mrr, avg_mrr, total_size = compute_loss_mrr(
                similarity_scores, batch_loss, batch_size, total_loss, total_mrr, total_size
            )

            if is_train:
                training_ctx.train_global_step += batch_size
            else:
                training_ctx.val_global_step += batch_size

            t_batch.set_postfix({f"{prefix}_loss": f"{per_sample_loss.item():10}"})
            t_batch.update(1)

            if batch_idx % log_interval == 0:
                if training_ctx.tensorboard is not None:
                    training_ctx.tensorboard.add_scalars(
                        {f"{prefix}_loss": avg_loss, f"{prefix}_mrr": avg_mrr},
                        group=prefix,
                        sub_group="batch",
                        global_step=training_ctx.train_global_step if is_train else training_ctx.val_global_step,
                    )
                if training_ctx.wandb_activated:
                    wandb.log(
                        {f"{prefix}_batch_loss": avg_loss, f"{prefix}_batch_mrr": avg_mrr},
                        step=training_ctx.train_global_step,
                    )
    used_time = UsedTime(time.time() - epoch_start)
    if training_ctx.tensorboard is not None:  # mypy needs that
        training_ctx.tensorboard.add_scalars(
            {
                f"{prefix}_loss": avg_loss,
                f"{prefix}_mrr": avg_mrr,
                f"{prefix}_samples_per_sec": int(total_size / used_time),
            },
            group=prefix,
            sub_group="epoch",
            global_step=epoch,
        )
    if training_ctx.wandb_activated:
        wandb.log(
            {
                f"epoch": epoch,
                f"{prefix}_epoch_loss": avg_loss,
                f"{prefix}_epoch_mrr": avg_mrr,
                f"{prefix}_epoch_samples_per_sec": int(total_size / used_time),
            }
        )

    return EpochResult(avg_loss, avg_mrr, used_time)


def run(args, tag_in_vcs=False) -> None:
    logger.debug("Building Training Context")
    training_ctx: CodeSearchTrainingContext
    print("args", args)
    conf: ConfigTree
    if args["--restore"] is not None and args["--config"] is not None:
        conf_file = args["--config"]
        logger.info(f"Config file: {conf_file}")

        conf = ConfigFactory.parse_file(conf_file)
        logger.info(f"Config {conf}")

        restore_dir = args["--restore"]

        logger.info(f"Build Training Context from config {conf_file} and dir {restore_dir}")
        training_ctx = CodeSearchTrainingContext.build_context_from_hocon_and_dir(conf, restore_dir)

    elif args["--restore"] is not None:
        conf = None
        restore_dir = args["--restore"]
        logger.info(f"Restoring Training Context from directory{restore_dir}")
        training_ctx = CodeSearchTrainingContext.build_context_from_dir(restore_dir)

    elif args["--config"] is not None:
        conf_file = args["--config"]
        logger.info(f"Config file: {conf_file}")

        conf = ConfigFactory.parse_file(conf_file)
        logger.info(f"Config {conf}")

        logger.info(f"Build Training Context from config {conf_file}")
        training_ctx = CodeSearchTrainingContext.build_context_from_hocon(conf)

    else:
        logger.error("need --config or --restore at least")
        sys.exit(1)

    # Build Train Dataloader
    if conf is None or not conf["training.short_circuit"]:
        train_dataset = training_ctx.build_lang_dataset(DatasetType.TRAIN)
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=training_ctx.train_batch_size,
            sampler=BalancedBatchSchedulerSampler(dataset=train_dataset, batch_size=training_ctx.train_batch_size),
        )
        logger.info(f"Built train_dataloader [Length:{len(train_dataloader)} x Batch:{training_ctx.train_batch_size}]")

    # Build Val Dataloader
    val_dataset = training_ctx.build_lang_dataset(DatasetType.VAL)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=training_ctx.val_batch_size,
        sampler=BalancedBatchSchedulerSampler(dataset=val_dataset, batch_size=training_ctx.val_batch_size),
    )
    logger.info(f"Built val_dataloader [Length:{len(val_dataloader)} x Batch:{training_ctx.val_batch_size}]")

    with trange(training_ctx.start_epoch, training_ctx.epochs) as t_epoch:
        # for epoch in range(start_epoch, epochs):
        for epoch in t_epoch:
            _ = run_epoch(
                prefix="train",
                epoch=epoch,
                training_ctx=training_ctx,
                dataloader=train_dataloader if (conf is None or not conf["training.short_circuit"]) else val_dataloader,
                log_interval=max(
                    int(
                        len(
                            train_dataloader if (conf is None or not conf["training.short_circuit"]) else val_dataloader
                        )
                        / 100
                    ),
                    training_ctx.min_log_interval,
                ),
                is_train=True,
            )

            val_result = run_epoch(
                prefix="val",
                epoch=epoch,
                training_ctx=training_ctx,
                dataloader=val_dataloader,
                log_interval=min(int(len(val_dataloader) / 10), training_ctx.min_log_interval),
                # tb=tb,
                is_train=False,
            )

            save_records_last(Path(training_ctx.output_dir) / training_ctx.training_full_name, training_ctx)

            if val_result.loss < training_ctx.best_loss:
                training_ctx.best_loss = val_result.loss
                training_ctx.best_loss_epoch = epoch
                logger.info(f"New best model loss:{training_ctx.best_loss} epoch:{training_ctx.best_loss_epoch}")
                save_records_best(
                    training_ctx.output_dir / training_ctx.training_full_name, training_ctx, suffix="loss"
                )

            if val_result.mrr > training_ctx.best_mrr:
                training_ctx.best_mrr = val_result.mrr
                training_ctx.best_mrr_epoch = epoch
                logger.info(f"New best model MRR:{training_ctx.best_mrr} epoch:{training_ctx.best_mrr_epoch}")
                save_records_best(training_ctx.output_dir / training_ctx.training_full_name, training_ctx, suffix="mrr")

            t_epoch.set_postfix(
                {
                    f"best_loss": training_ctx.best_loss,
                    "best_loss_epoch": training_ctx.best_loss_epoch,
                    "best_mrr": training_ctx.best_mrr,
                    "best_mrr_epoch": training_ctx.best_mrr_epoch,
                }
            )
            t_epoch.update(1)


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args["--debug"])
