#!/usr/bin/env python3
"""
Usage:
    tokenizers_build.py [options]
    tokenizers_build.py [options]

Options:
    -h --help                        Show this screen.
    --config FILE                    Specify HOCON config file. [default: ./conf/default.conf]
    --debug                          Enable debug routines. [default: False]
"""


from docopt import docopt
from loguru import logger
import sys
import torch
from dpu_utils.utils import run_and_debug
from pyhocon import ConfigFactory, ConfigTree
from torch.utils.data import DataLoader
from codenets.codesearchnet.dataset import (
    DatasetParams,
    BalancedBatchSchedulerSampler,
    build_lang_dataset,
    build_lang_dataset_single_code_tokenizer,
)
from codenets.codesearchnet.tokenizer_recs import (
    load_query_code_tokenizers_from_hocon,
    build_or_load_original_tokenizers,
    load_query_code_tokenizers_from_hocon_single_code_tokenizer,
)
from codenets.utils import expand_data_path


print("Torch version", torch.__version__)  # type: ignore

logger.remove()
logger.add(sys.stderr, level="DEBUG", colorize=True, backtrace=False)


def run_original(args, tag_in_vcs=False) -> None:
    conf_file = args["--config"]
    logger.info(f"config file {conf_file}")

    conf: ConfigTree = ConfigFactory.parse_file(conf_file)
    logger.info(f"config {conf}")

    train_data_params = DatasetParams(**conf["dataset.train.params"])
    val_data_params = DatasetParams(**conf["dataset.val.params"])
    query_tokenizer, code_tokenizers = build_or_load_original_tokenizers(
        dirs=expand_data_path(conf["dataset.train.dirs"]),
        name="train",
        data_params=train_data_params,
        pickle_path=conf["training.pickle_path"],
    )

    train_dataset = build_lang_dataset(
        expand_data_path(conf["dataset.train.dirs"]),
        "train",
        train_data_params,
        query_tokenizer,
        code_tokenizers,
        pickle_path=conf["training.pickle_path"],
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=conf["training.batch_size.train"],
        sampler=BalancedBatchSchedulerSampler(dataset=train_dataset, batch_size=conf["training.batch_size.train"]),
    )
    logger.info(f"train_dataloader [{len(train_dataloader)} samples]")

    val_dataset = build_lang_dataset(
        expand_data_path(conf["dataset.val.dirs"]),
        "val",
        val_data_params,
        query_tokenizer,
        code_tokenizers,
        pickle_path=conf["training.pickle_path"],
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=conf["training.batch_size.val"],
        sampler=BalancedBatchSchedulerSampler(dataset=val_dataset, batch_size=conf["training.batch_size.val"]),
    )
    logger.info(f"val_dataloader [{len(val_dataloader)} samples]")


def run_single_code_tokenizer(args, tag_in_vcs=False) -> None:
    conf_file = args["--config"]
    logger.info(f"config file {conf_file}")

    conf: ConfigTree = ConfigFactory.parse_file(conf_file)
    logger.info(f"config {conf}")

    res = load_query_code_tokenizers_from_hocon_single_code_tokenizer(conf)
    if res is not None:
        query_tokenizer, code_tokenizer = res

    train_data_params = DatasetParams(**conf["dataset.train.params"])
    train_dataset = build_lang_dataset_single_code_tokenizer(
        expand_data_path(conf["dataset.train.dirs"]),
        "train_single",
        train_data_params,
        query_tokenizer,
        code_tokenizer,
        lang_token="<lg>",
        pickle_path=conf["training.pickle_path"],
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=conf["training.batch_size.train"],
        sampler=BalancedBatchSchedulerSampler(dataset=train_dataset, batch_size=conf["training.batch_size.train"]),
    )
    logger.info(f"train_dataloader [{len(train_dataloader)} samples]")

    val_data_params = DatasetParams(**conf["dataset.val.params"])
    val_dataset = build_lang_dataset_single_code_tokenizer(
        expand_data_path(conf["dataset.val.dirs"]),
        "val_single",
        val_data_params,
        query_tokenizer,
        code_tokenizer,
        lang_token="<lg>",
        pickle_path=conf["training.pickle_path"],
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=conf["training.batch_size.val"],
        sampler=BalancedBatchSchedulerSampler(dataset=val_dataset, batch_size=conf["training.batch_size.val"]),
    )
    logger.info(f"val_dataloader [{len(val_dataloader)} samples]")
    logger.info(list(train_dataloader)[:5])


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run_single_code_tokenizer(args), args["--debug"])
