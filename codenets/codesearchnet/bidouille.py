#!/usr/bin/env python3
"""
Usage:
    bidouille.py [options]
    bidouille.py [options]

Options:
    -h --help                        Show this screen.
    --config FILE                    Specify HOCON config file. [default: ./conf/default.conf]
    --debug                          Enable debug routines. [default: False]
"""

from docopt import docopt

from loguru import logger
import sys

import torch

# from torch_metadata import expand_data_path, DataParams, load_data_from_dirs
# from torch_data import load_tokenizers, TokenizerFn, BalancedBatchSchedulerSampler, build_lang_dataset, DataParams

# from torch_metadata import DataParams
# from torch_utils import expand_data_path
from dpu_utils.utils import run_and_debug

# from torch_model import CodeSearchBaseModel, codesearchmodel_sbert_from_hocon
# from torch_pooler import MeanWeightedPooler

# from torch_loss import SoftmaxCrossEntropyLossAndSimilarityScore
# from torch_utils import expand_data_path

# from torch import nn

# from tensorboard_utils import Tensorboard
# from torch_model import RecordableModule, recordable_module, recordable_tokenizer, recordable_dict, recordable_state
# from torch_loss import load_loss_and_similarity_function
# from torch_save import save_records, recover_records
from pyhocon import ConfigFactory, ConfigTree
from codenets.codesearchnet.multi_branch_model import multibranch_bert_from_hocon, MultiBranchCodeSearchModel

print("Torch version", torch.__version__)  # type: ignore

logger.remove()
logger.add(sys.stderr, level="DEBUG", colorize=True, backtrace=False)


def run(args, tag_in_vcs=False) -> None:
    conf_file = args["--config"]
    logger.info(f"config file {conf_file}")

    conf: ConfigTree = ConfigFactory.parse_file(conf_file)
    logger.info(f"config {conf}")

    model = multibranch_bert_from_hocon(conf)

    model.save("./tests")

    model_reloaded = MultiBranchCodeSearchModel.load("./tests")


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args["--debug"])
