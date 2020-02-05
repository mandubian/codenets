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
from codenets.codesearchnet.dataset import DatasetParams
from codenets.codesearchnet.tokenizer_recs import build_or_load_original_tokenizers
from codenets.utils import expand_data_path


print("Torch version", torch.__version__)  # type: ignore

logger.remove()
logger.add(sys.stderr, level="DEBUG", colorize=True, backtrace=False)


def run(args, tag_in_vcs=False) -> None:
    conf_file = args["--config"]
    logger.info(f"config file {conf_file}")

    conf: ConfigTree = ConfigFactory.parse_file(conf_file)
    logger.info(f"config {conf}")

    train_data_params = DatasetParams(**conf["dataset.train.params"])
    query_tokenizer, code_tokenizers = build_or_load_original_tokenizers(
        dirs=expand_data_path(conf["dataset.train.dirs"]),
        name="train",
        data_params=train_data_params,
        pickle_path=conf["training.pickle_path"],
        force_rebuild=True,
    )


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args["--debug"])
