#!/usr/bin/env python3
"""
Main to test dataset loading

Usage:
    dataset_main.py [options]
    dataset_main.py [options]

Options:
    -h --help                        Show this screen.
    --config FILE                    Specify HOCON config file. [default: ./conf/default.conf]
    --debug                          Enable debug routines. [default: False]
"""


from docopt import docopt
from loguru import logger
import sys
import torch
import itertools
from dpu_utils.utils import run_and_debug
from pyhocon import ConfigFactory, ConfigTree
from torch.utils.data import DataLoader
from codenets.codesearchnet.dataset_utils import BalancedBatchSchedulerSampler, DatasetType
from codenets.codesearchnet.training_ctx import CodeSearchTrainingContext
from codenets.utils import expand_data_path


print("Torch version", torch.__version__)  # type: ignore

logger.remove()
logger.add(sys.stderr, level="DEBUG", colorize=True, backtrace=False)


def run(args, tag_in_vcs=False) -> None:
    conf_file = args["--config"]
    logger.info(f"config file {conf_file}")

    conf: ConfigTree = ConfigFactory.parse_file(conf_file)
    logger.info(f"config {conf}")
    logger.info(f"Build Training Context from config {conf_file}")
    training_ctx = CodeSearchTrainingContext.build_context_from_hocon(conf)

    train_dataset = training_ctx.build_lang_dataset(DatasetType.TRAIN)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=conf["training.batch_size.train"],
        sampler=BalancedBatchSchedulerSampler(dataset=train_dataset, batch_size=conf["training.batch_size.train"]),
    )
    logger.info(f"train_dataloader [{len(train_dataloader)} samples]")

    for batch in itertools.islice(train_dataloader, 5):
        logger.info(f"batch {batch}")

    # val_dataset = training_ctx.build_lang_dataset(DatasetType.VAL)
    # val_dataloader = DataLoader(
    #     dataset=val_dataset,
    #     batch_size=conf["training.batch_size.val"],
    #     sampler=BalancedBatchSchedulerSampler(dataset=val_dataset, batch_size=conf["training.batch_size.val"]),
    # )
    # logger.info(f"val_dataloader [{len(val_dataloader)} samples]")

    # for batch in itertools.islice(val_dataloader, 5):
    #     logger.info(f"batch {batch}")


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args["--debug"])
