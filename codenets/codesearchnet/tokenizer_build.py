#!/usr/bin/env python3
"""
Usage:
    tokenizers_huggingface_build.py [options]
    tokenizers_huggingface_build.py [options]

Options:
    -h --help                        Show this screen.
    --config FILE                    Specify HOCON config file. [default: ./conf/default.conf]
    --debug                          Enable debug routines. [default: False]
"""


from docopt import docopt
from loguru import logger
import sys
import torch
from typing import List
from dpu_utils.utils import run_and_debug
from pyhocon import ConfigFactory, ConfigTree

from codenets.codesearchnet.training_ctx import CodeSearchTrainingContext, default_sample_update
from codenets.codesearchnet.dataset_utils import DatasetType
from codenets.codesearchnet.tokenizer_recs import build_most_common_tokens

print("Torch version", torch.__version__)  # type: ignore

logger.remove()
logger.add(sys.stderr, level="DEBUG", colorize=True, backtrace=False)


def run_single_code_tokenizer(args, tag_in_vcs=False) -> None:
    conf_file = args["--config"]
    logger.info(f"config file {conf_file}")

    conf: ConfigTree = ConfigFactory.parse_file(conf_file)
    logger.info(f"config {conf}")

    logger.info(f"Build Training Context from config {conf_file}")
    training_ctx = CodeSearchTrainingContext.build_context_from_hocon(conf)

    # training_ctx.build_tokenizers(from_dataset_type=DatasetType.TRAIN)

    # txt = "python <lg> def toto():"
    # logger.info("encoded", training_ctx.tokenize_code_sentences([txt]))
    # txt = "go <lg> function getCounts() { return 0 }"
    # logger.info("encoded", training_ctx.tokenize_code_sentences([txt]))

    most_commons = build_most_common_tokens(
        training_ctx.train_dirs, training_ctx.train_data_params, training_ctx.tokenizers_build_path
    )
    logger.info(f"most_commons {most_commons}")


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run_single_code_tokenizer(args), args["--debug"])
