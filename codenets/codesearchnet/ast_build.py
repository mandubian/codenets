#!/usr/bin/env python3
"""
Usage:
    eval.py [options] SAVE_FOLDER TRAIN_DATA_PATH VALID_DATA_PATH TEST_DATA_PATH
    eval.py [options] [SAVE_FOLDER]

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
    --config FILE                    Specify HOCON config file.
    --debug                          Enable debug routines. [default: False]
"""

from typing import Dict, List, Optional, Tuple, Set
from dpu_utils.utils import run_and_debug
from docopt import docopt
from loguru import logger
import itertools
import os
import pickle
from torch.utils.data import DataLoader
from pathlib import Path
from pyhocon import ConfigFactory
from torch import nn
from torch import Tensor
import torch
import numpy as np
import pandas as pd

from tree_sitter import Language, Parser, Node
from codenets.codesearchnet.copied_code.utils import read_file_samples
from sklearn.metrics.pairwise import pairwise_distances
from codenets.codesearchnet.dataset_utils import BalancedBatchSchedulerSampler, DatasetType
from codenets.codesearchnet.data import DatasetParams
from codenets.codesearchnet.training_ctx import CodeSearchTrainingContext
from codenets.codesearchnet.query_code_siamese.dataset import load_data_from_dirs
from codenets.codesearchnet.code_ast.ast_utils import build_language_ast

"""Evaluating SBert."""


def run(args, tag_in_vcs=False) -> None:
    os.environ["WANDB_MODE"] = "dryrun"

    logger.debug("Building Training Context")
    conf_file = args["--config"]
    conf = ConfigFactory.parse_file(conf_file)

    logger.info(f"Restoring Training Context from config {conf_file}")
    training_ctx = CodeSearchTrainingContext.build_context_from_hocon(conf)

    # dirs = [Path("/home/mandubian/workspaces/tools/CodeSearchNet/resources/data/ruby/final/jsonl/valid/")]
    # build_language_ast("val", training_ctx.val_dirs, training_ctx.pickle_path, training_ctx.val_data_params)
    # build_language_ast("train", training_ctx.train_dirs, training_ctx.pickle_path, training_ctx.train_data_params)
    build_language_ast("test", training_ctx.test_dirs, training_ctx.pickle_path, training_ctx.test_data_params)

    # Language.build_library(
    #     # Store the library in the `build` directory
    #     "build/my-languages.so",
    #     # Include one or more languages
    #     [
    #         "vendor/tree-sitter-go",
    #         "vendor/tree-sitter-java",
    #         "vendor/tree-sitter-javascript",
    #         "vendor/tree-sitter-python",
    #         "vendor/tree-sitter-php",
    #         "vendor/tree-sitter-ruby",
    #     ],
    # )

    # parser = Parser()

    # code_php = """
    # <?php
    # protected function checkAndSetAuthentication($repositoryName, $username, $password = null)
    #     {
    #         if ($this->hasAuthentication($repositoryName)) {
    #             $auth = $this->getAuthentication($repositoryName);
    #             if ($auth['username'] === $username && $auth['password'] === $password) {
    #                 return;
    #             }

    #             $this->writeError(
    #                 sprintf(
    #                     "<warning>Warning: You should avoid overwriting already defined auth settings for %s.</warning>",
    #                     $repositoryName
    #                 )
    #             );
    #         }
    #         $this->setAuthentication($repositoryName, $username, $password);
    #     }
    # ?>
    # """
    # PHP_LANGUAGE = Language("build/my-languages.so", "php")
    # parser.set_language(PHP_LANGUAGE)
    # tree = parser.parse(bytes(code_php, "utf8"))
    # cursor = tree.walk()
    # print(cursor.node.sexp())

    # skip_node_types = ["ERROR", "<?php", "?>"]
    # all_tokens_php, special_tokens_php = breadth_first_path("php", code_php, cursor, skip_node_types=skip_node_types)
    # print("all_tokens_php", all_tokens_php)
    # print("special_tokens_php", special_tokens_php)

    # JAVA_LANGUAGE = Language("build/my-languages.so", "java")
    # # parser = Parser()
    # parser.set_language(JAVA_LANGUAGE)
    # code_java = """
    # class A {
    #     public int b() {
    #         int c = 5;
    #     }
    # }
    # """
    # tree = parser.parse(bytes(code_java, "utf8"))
    # cursor = tree.walk()
    # print("code_java", code_java)
    # print(cursor.node.sexp())
    # all_tokens_java, special_tokens_java = breadth_first_path(code_java, cursor)
    # print("all_tokens_java", all_tokens_java)
    # print("special_tokens_java", special_tokens_java)

    # print("===================================================")

    # PY_LANGUAGE = Language("build/my-languages.so", "python")
    # parser.set_language(PY_LANGUAGE)
    # code_python = """
    # def foo():
    #     if bar:
    #         a: List[str] = ["toto", "tata"]
    #         baz(a, b, 5)
    # """
    # tree = parser.parse(bytes(code_python, "utf8"))
    # cursor = tree.walk()
    # print("code_python", code_python)
    # print(cursor.node.sexp())
    # all_tokens_python, special_tokens_python = breadth_first_path(code_python, cursor)
    # print("all_tokeall_tokens_pythonns", all_tokens_python)
    # print("special_tokens_python", special_tokens_python)

    # special_tokens = special_tokens_python.union(special_tokens_java)
    # print("special_tokens", special_tokens)
    # training_ctx.tokenizer.vocab.add_special_tokens(list(special_tokens))

    # print("JAVA", training_ctx.tokenize_code_sentences([" ".join(all_tokens_java)], max_length=256))
    # print("PYTHON", training_ctx.tokenize_code_sentences([" ".join(all_tokens_python)], max_length=256))


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args["--debug"])
