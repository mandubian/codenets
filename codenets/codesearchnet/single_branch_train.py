#!/usr/bin/env python3
"""
Usage:
    multi_branch_train.py [options]
    multi_branch_train.py [options]

*_DATA_PATH arguments may either accept (1) directory filled with .jsonl.gz files that we use as data,
or a (2) plain text file containing a list of such directories (used for multi-language training).

In the case that you supply a (2) plain text file, all directory names must be separated by a newline.
For example, if you want to read from multiple directories you might have a plain text file called
data_dirs_train.txt with the below contents:

> cat ~/src/data_dirs_train.txt
azure://semanticcodesearch/pythondata/Processed_Data/jsonl/train
azure://semanticcodesearch/csharpdata/split/csharpCrawl-train

Options:
    -h --help                        Show this screen.
    --config FILE                    Specify HOCON config file. [default: ../conf/default.conf]
    --restore DIR                    specify restoration dir. [optional]
    --debug                          Enable debug routines. [default: False]
"""

import os
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

from codenets.codesearchnet.dataset import BalancedBatchSchedulerSampler, build_lang_dataset_single_code_tokenizer
from codenets.codesearchnet.single_branch_model import SingleBranchTrainingContext
from codenets.save import save_records_best, save_records_last

logger.remove()
logger.add(sys.stderr, level="DEBUG", colorize=True, backtrace=False)


@dataclass
class EpochResult:
    loss: float
    mrr: float
    used_time: float


def run_epoch(
    prefix: str,
    epoch: int,
    training_ctx: SingleBranchTrainingContext,
    dataloader: torch.utils.data.DataLoader,
    log_interval: int,
    # tb: Tensorboard = None,
    is_train: bool = True,
) -> EpochResult:
    """
    Run epoch computation according to is_train flag
    Please note that IT MUTATES training_ctx
    (We need to modify models & optimizer in all cases so let's mutate)
    """
    epoch_loss = 0.0
    epoch_samples = 0
    mrr_sum = 0.0
    training_ctx.epoch = epoch
    epoch_start = time.time()
    if is_train:
        logger.debug("Train Mode")
        training_ctx.model.train()
        torch.set_grad_enabled(True)
    else:
        logger.debug("Eval Mode")
        training_ctx.model.eval()
        torch.set_grad_enabled(False)

    training_ctx.model.zero_grad()
    with tqdm(total=len(dataloader)) as t_batch:
        for batch_idx, batch in enumerate(dataloader):
            languages, similarity, query_tokens, query_tokens_mask, code_tokens, code_tokens_mask = [
                t.to(training_ctx.device) for t in batch
            ]
            # if is_train:
            #     optimizer.zero_grad()

            (query_embedding, code_embedding) = training_ctx.model(
                languages=languages,
                query_tokens=query_tokens,
                query_tokens_mask=query_tokens_mask,
                code_tokens=code_tokens,
                code_tokens_mask=code_tokens_mask,
            )
            per_sample_losses, similarity_scores = training_ctx.losses_scores_fn(
                query_embedding, code_embedding, similarity
            )
            per_sample_loss = torch.mean(per_sample_losses)

            if is_train:
                per_sample_loss.backward()
                training_ctx.optimizer.step()
                training_ctx.model.zero_grad()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            nb_samples = batch[0].size()[0]

            # compute MRR
            # extract the logits from the diagonal of the matrix, which are the logits corresponding to the ground-truth
            correct_scores = similarity_scores.diagonal()
            # compute how many queries have bigger logits than the ground truth (the diagonal)
            # the elements that are incorrectly ranked
            compared_scores = similarity_scores.ge(correct_scores.unsqueeze(dim=-1)).float()
            compared_scores_nb = torch.sum(compared_scores, dim=1)
            # logger.debug(f"compared_scores_nb {compared_scores_nb}")
            per_sample_mrr = torch.div(1.0, compared_scores_nb)
            per_batch_mrr = torch.sum(per_sample_mrr) / nb_samples
            # logger.debug(f"per_sample_mrr {per_sample_mrr.cpu().numpy()}")

            if is_train:
                training_ctx.train_global_step += nb_samples
            else:
                training_ctx.val_global_step += nb_samples
            epoch_samples += nb_samples
            epoch_loss += per_sample_loss.item() * nb_samples
            loss = epoch_loss / max(1, epoch_samples)

            mrr_sum += per_batch_mrr.item() * nb_samples
            mrr = mrr_sum / max(1, epoch_samples)

            t_batch.set_postfix({f"{prefix}_loss": f"{per_sample_loss.item():10}"})
            t_batch.update(1)

            if batch_idx % log_interval == 0:
                if training_ctx.tensorboard is not None:
                    training_ctx.tensorboard.add_scalars(
                        {f"{prefix}_loss": loss, f"{prefix}_mrr": mrr},
                        group=prefix,
                        sub_group="batch",
                        global_step=training_ctx.train_global_step if is_train else training_ctx.val_global_step,
                    )
                if training_ctx.wandb_activated:
                    wandb.log(
                        {f"{prefix}_batch_loss": loss, f"{prefix}_batch_mrr": mrr}, step=training_ctx.train_global_step
                    )
    used_time = time.time() - epoch_start
    if training_ctx.tensorboard is not None:  # mypy needs that
        training_ctx.tensorboard.add_scalars(
            {f"{prefix}_loss": loss, f"{prefix}_mrr": mrr, f"{prefix}_samples_per_sec": int(epoch_samples / used_time)},
            group=prefix,
            sub_group="epoch",
            global_step=epoch,
        )
    if training_ctx.wandb_activated:
        wandb.log(
            {
                f"epoch": epoch,
                f"{prefix}_epoch_loss": loss,
                f"{prefix}_epoch_mrr": mrr,
                f"{prefix}_epoch_samples_per_sec": int(epoch_samples / used_time),
            }
        )

    return EpochResult(loss, mrr, used_time)


def run(args, tag_in_vcs=False) -> None:
    logger.debug("Building Training Context")
    training_ctx: SingleBranchTrainingContext
    print("args", args)
    if args["--restore"] is not None:
        restore_dir = args["--restore"]
        logger.info(f"Restoring Training Context from directory{restore_dir}")
        training_ctx = SingleBranchTrainingContext.load(restore_dir)
    else:
        conf_file = args["--config"]
        logger.info(f"Config file: {conf_file}")

        conf: ConfigTree = ConfigFactory.parse_file(conf_file)
        logger.info(f"Config {conf}")

        logger.info(f"Build Training Context from config {conf_file}")
        training_ctx = SingleBranchTrainingContext.from_hocon(conf)

    # Build Train Dataloader
    train_dataset = build_lang_dataset_single_code_tokenizer(
        training_ctx.train_dirs,
        "train",
        training_ctx.train_data_params,
        training_ctx.query_tokenizer,
        training_ctx.code_tokenizer,
        lang_token="<lg>",
        pickle_path=training_ctx.pickle_path,
        parallelize=training_ctx.train_data_params.parallelize,
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=training_ctx.train_batch_size,
        sampler=BalancedBatchSchedulerSampler(dataset=train_dataset, batch_size=training_ctx.train_batch_size),
    )
    logger.info(f"Built train_dataloader [Length:{len(train_dataloader)} x Batch:{training_ctx.train_batch_size}]")

    # Build Val Dataloader
    val_dataset = build_lang_dataset_single_code_tokenizer(
        training_ctx.val_dirs,
        "val",
        training_ctx.val_data_params,
        training_ctx.query_tokenizer,
        training_ctx.code_tokenizer,
        lang_token="<lg>",
        pickle_path=training_ctx.pickle_path,
        parallelize=training_ctx.train_data_params.parallelize,
    )
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
                dataloader=train_dataloader,
                # dataloader=val_dataloader,
                log_interval=max(int(len(train_dataloader) / 100), training_ctx.min_log_interval),
                # log_interval=max(int(len(val_dataloader) / 100), training_ctx.min_log_interval),
                # tb=tb,
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
                training_ctx.best_mrr = val_result.mrr
                training_ctx.best_epoch = epoch
                logger.info(
                    f"New best model loss:{training_ctx.best_loss} best_mrr:{training_ctx.best_mrr} epoch:{training_ctx.best_epoch}"
                )
                save_records_best(training_ctx.output_dir / training_ctx.training_full_name, training_ctx)

            t_epoch.set_postfix(
                {
                    f"best_loss": training_ctx.best_loss,
                    "best_mrr": training_ctx.best_mrr,
                    "best_epoch": training_ctx.best_epoch,
                }
            )
            t_epoch.update(1)


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args["--debug"])
