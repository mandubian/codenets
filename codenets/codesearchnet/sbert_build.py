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

from typing import Dict, List
from sentence_transformers import SentenceTransformer
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

from tree_sitter import Language, Parser
from codenets.codesearchnet.copied_code.utils import read_file_samples
from sklearn.metrics.pairwise import pairwise_distances
from codenets.codesearchnet.dataset_utils import BalancedBatchSchedulerSampler, DatasetType
from codenets.codesearchnet.data import DatasetParams
from codenets.codesearchnet.training_ctx import CodeSearchTrainingContext
from codenets.codesearchnet.query_code_siamese.dataset import load_data_from_dirs

"""Evaluating SBert."""


def run(args, tag_in_vcs=False) -> None:
    # os.environ["WANDB_MODE"] = "dryrun"

    logger.debug("Building Training Context")
    conf_file = args["--config"]
    conf = ConfigFactory.parse_file(conf_file)

    logger.info(f"Restoring Training Context from config {conf_file}")
    training_ctx = CodeSearchTrainingContext.build_context_from_hocon(conf)

    # val_dataset = training_ctx.build_lang_dataset(DatasetType.VAL)
    # if val_dataset.collate_fn is not None:
    #     val_dataloader = DataLoader(
    #         dataset=val_dataset,
    #         batch_size=conf["training.batch_size.val"],
    #         sampler=BalancedBatchSchedulerSampler(dataset=val_dataset, batch_size=conf["training.batch_size.val"]),
    #         collate_fn=val_dataset.collate_fn,
    #     )
    # else:
    #     val_dataloader = DataLoader(
    #         dataset=val_dataset,
    #         batch_size=conf["training.batch_size.val"],
    #         sampler=BalancedBatchSchedulerSampler(dataset=val_dataset, batch_size=conf["training.batch_size.val"]),
    #     )

    val_dataloader = training_ctx.build_lang_dataloader(DatasetType.VAL)
    logger.info(f"val_dataloader [{len(val_dataloader)} samples]")

    # train_dataloader = training_ctx.build_lang_dataloader(DatasetType.TRAIN)
    # logger.info(f"train_dataloader [{len(train_dataloader)} samples]")

    # df = pd.read_parquet("./pickles/train_qc_30k_embeddings.parquet")
    # print(df.info())

    # z = df.iloc[0][0]
    # print("z", z.shape)
    from annoy import AnnoyIndex

    t = AnnoyIndex(768, "angular")
    # for index, row in df.iterrows():
    #     print(row.shape)
    #     t.add_item(index, row[0])
    # t.build(10)  # 10 trees
    # t.save("./pickles/train_qc_30k_embeddings.ann")

    t.load("./pickles/val_qc_30k_embeddings.ann")

    # for i in range(0, 100):
    #     print(i, 99, 1.0 - t.get_distance(i, 99))

    for batch in val_dataloader:  # itertools.islice(val_dataloader, 0, 1000):
        indices, languages, similarity, query_tokens, query_tokens_mask, code_tokens, code_tokens_mask, code_lang_weights = (
            batch
        )
        toks = [toks.cpu().numpy()[: len(mask[mask != 0])] for (toks, mask) in zip(query_tokens, query_tokens_mask)]
        toks = training_ctx.decode_query_tokens(toks)
        qs = [str((t, score)) for (t, score) in list(zip(toks, similarity))]
        for i, scores in enumerate(similarity):
            for j, s in enumerate(scores):
                if s > 0.5 and i != j:
                    print(s, toks[i], toks[j])

        # print("query", "\n".join(qs))

    #     # print("query_tokens", query_tokens)
    #     # 5 for removing "<qy> "
    #     toks = [toks.cpu().numpy()[: len(mask[mask != 0])] for (toks, mask) in zip(query_tokens, query_tokens_mask)]
    #     toks = training_ctx.decode_query_tokens(toks)
    #     # print("toks", toks)
    #     qs = [str((t, score)) for (t, score) in list(zip(toks, similarity))]
    #     print("query", "\n".join(qs))
    #     print("-----------")

    # data_file = (
    #     "/home/mandubian/workspaces/tools/CodeSearchNet/resources/data/python/final/jsonl/valid/python_valid_0.jsonl.gz"
    # )
    # filename = os.path.basename(data_file)
    # file_language = filename.split("_")[0]

    # samples = list(read_file_samples(data_file))

    # sample0 = samples[0]
    # sample1 = samples[1]
    # logger.info(f"keys {sample0.keys()}")
    # logger.info(f"sample docstring {sample0['docstring_tokens']}")
    # query0 = " ".join(samples[0]["docstring_tokens"])
    # logger.info(f"query0 {query0}")
    # query_embeddings0 = model.encode([query0])
    # # logger.info(f"query_embeddings0 {query_embeddings0}")
    # query1 = " ".join(sample1["docstring_tokens"])
    # query_embeddings1 = model.encode([query1])

    # distances = pairwise_distances(query_embeddings0, query_embeddings1, metric="cosine")
    # logger.info(f"distances {distances}")

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
    # PY_LANGUAGE = Language("build/my-languages.so", "python")
    # parser = Parser()
    # parser.set_language(PY_LANGUAGE)
    # tree = parser.parse(bytes(samples[0]["code"], "utf8"))

    # logger.info(f"tree {tree}")


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args["--debug"])
