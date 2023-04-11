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
from dpu_utils.utils import run_and_debug
from pyhocon import ConfigFactory, ConfigTree

from codenets.codesearchnet.training_ctx import CodeSearchTrainingContext
from codenets.codesearchnet.tokenizer_recs import build_most_common_tokens

print("Torch version", torch.__version__)

logger.remove()
logger.add(sys.stderr, level="DEBUG", colorize=True, backtrace=False)


def run(args, tag_in_vcs=False) -> None:
    conf_file = args["--config"]
    logger.info(f"config file {conf_file}")

    conf: ConfigTree = ConfigFactory.parse_file(conf_file)
    logger.info(f"config {conf}")

    # logger.info(f"Build Training Context from config {conf_file}")
    # training_ctx = CodeSearchTrainingContext.build_context_from_hocon(conf)

    # training_ctx.build_tokenizers(from_dataset_type=DatasetType.TRAIN)

    logger.info(f"Reload Training Context from config {conf_file} with built tokenizers")
    training_ctx = CodeSearchTrainingContext.build_context_from_hocon(conf)

    txt = "python <lg> def toto():"
    logger.info(f"encoded {training_ctx.tokenize_code_sentences([txt])}")
    txt = "go <lg> function getCounts() { return 0 }"
    logger.info(f"encoded {training_ctx.tokenize_code_sentences([txt])}")

    most_commons = build_most_common_tokens(
        training_ctx.train_dirs, training_ctx.train_data_params, training_ctx.tokenizers_build_path,
        parallelize=False
    )
    logger.info(f"most_commons {most_commons}")


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args["--debug"])
