#!/usr/bin/env python3
"""
Usage:
    train.py [options] SAVE_FOLDER TRAIN_DATA_PATH VALID_DATA_PATH TEST_DATA_PATH
    train.py [options] [SAVE_FOLDER]

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
    --restore DIR                    specify restoration dir. [optional]
    --debug                          Enable debug routines. [default: False]
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict
import torch
import numpy as np
from docopt import docopt
from dpu_utils.utils import run_and_debug
from loguru import logger
import pandas as pd
from annoy import AnnoyIndex
import pickle
from tqdm import tqdm

from codenets.codesearchnet.multi_branch_model import MultiBranchTrainingContext


def get_language_defs(
    language: str,
    training_ctx: MultiBranchTrainingContext,  # per_code_language_tokenizers: Dict[str, TokenizerRecordable]
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
    pkl_file = training_ctx.pickle_path / f"{language}_dedupe_definitions_tokens_masks_v2.p"
    root_data_path = Path(training_ctx.conf["dataset.root_dir"])
    if not os.path.exists(pkl_file):
        logger.info("Evaluating language: %s" % language)
        definitions = pickle.load(open(root_data_path / "data/{}_dedupe_definitions_v2.pkl".format(language), "rb"))
        # codes_encoded = []
        # codes_masks = []
        function_tokens = [d["function_tokens"] for d in definitions]
        codes_encoded, codes_masks = training_ctx.code_tokenizers[language].encode_tokens(
            function_tokens, max_length=200
        )

        pickle.dump((codes_encoded, codes_masks), open(pkl_file, "wb"))
    else:
        definitions = pickle.load(open(root_data_path / "data/{}_dedupe_definitions_v2.pkl".format(language), "rb"))
        (codes_encoded, codes_masks) = pickle.load(open(pkl_file, "rb"))

    return (codes_encoded, codes_masks, definitions)


def compute_code_embeddings(
    language: str,
    training_ctx: MultiBranchTrainingContext,
    codes_encoded: List[np.ndarray],
    codes_masks: List[np.ndarray],
) -> List[np.ndarray]:
    lang_id = training_ctx.train_data_params.lang_ids[language]
    pkl_file = training_ctx.pickle_path / f"{language}_code_embeddings.p"
    if not os.path.exists(pkl_file):
        seg_length = 200
        segments = [
            (codes_encoded[x : x + seg_length], codes_masks[x : x + seg_length])
            for x in range(0, len(codes_encoded), seg_length)
        ]

        code_embeddings = []
        for (codes_enc, codes_m) in tqdm(segments):
            codes_encoded_t = torch.tensor(codes_enc, dtype=torch.long).to(training_ctx.device)
            codes_masks_t = torch.tensor(codes_m, dtype=torch.long).to(training_ctx.device)
            code_embeddings.append(
                training_ctx.model.encode_code(
                    lang_id=lang_id, code_tokens=codes_encoded_t, code_tokens_mask=codes_masks_t
                )
                .cpu()
                .numpy()
            )
        pickle.dump(code_embeddings, open(pkl_file, "wb"))
        return code_embeddings
    else:
        return pickle.load(open(pkl_file, "rb"))


def run(args, tag_in_vcs=False) -> None:
    logger.debug("Building Training Context")
    training_ctx: MultiBranchTrainingContext
    restore_dir = args["--restore"]
    logger.info(f"Restoring Training Context from directory{restore_dir}")
    training_ctx = MultiBranchTrainingContext.load(restore_dir)

    queries = pd.read_csv(training_ctx.queries_file)
    queries = list(queries["query"].values)
    queries_tokens, queries_masks = training_ctx.query_tokenizer.encode_sentences(
        queries, max_length=training_ctx.conf["dataset.common_params.query_max_num_tokens"]
    )
    logger.info(f"queries: {queries}")

    with torch.no_grad():
        query_embeddings = (
            training_ctx.model.encode_query(
                query_tokens=torch.tensor(queries_tokens, dtype=torch.long).to(training_ctx.device),
                query_tokens_mask=torch.tensor(queries_masks, dtype=torch.long).to(training_ctx.device),
            )
            .cpu()
            .numpy()
        )
        logger.info(f"query_embeddings: {query_embeddings.shape}")

        topk = 100
        predictions = []
        for language in ("python",):  # , "go", "javascript", "java", "php", "ruby"):
            (codes_encoded, codes_masks, definitions) = get_language_defs(language, training_ctx)

            code_embeddings = compute_code_embeddings(language, training_ctx, codes_encoded, codes_masks)

            # query_e = query_embeddings[0]
            # code_e = code_embeddings[0]

            # for code_e in code_embeddings[0]:
            # a = torch.tensor(code_embeddings[0][0], dtype=torch.float32)
            # b = torch.tensor(code_embeddings[1][0], dtype=torch.float32)

            # logger.debug(f"query_e_t {query_e_t}")
            # logger.debug(f"code_e_t {code_e_t}")
            # logger.info(f"a {a}")
            # logger.info(f"b {b}")
            # logger.info(f"sim {torch.cosine_similarity(a.unsqueeze(dim=0), b.unsqueeze(dim=0))}")
            indices: AnnoyIndex = AnnoyIndex(code_embeddings[0][0].shape[0], "angular")
            idx = 0
            for index, vectors in tqdm(enumerate(code_embeddings)):
                for vector in vectors:
                    if vector is not None:
                        if idx < 10:
                            logger.debug(f"vector {vector}")
                        indices.add_item(idx, vector)
                        idx += 1
            indices.build(10)

            for i, (query, query_embedding) in enumerate(zip(queries, query_embeddings)):
                idxs, distances = indices.get_nns_by_vector(query_embedding, topk, include_distances=True)
                if i < 10:
                    logger.debug(f"query_embedding {query_embedding}")
                    logger.debug(f"idxs:{idxs}, distances:{distances}")
                for idx2, _ in zip(idxs, distances):
                    predictions.append((query, language, definitions[idx2]["identifier"], definitions[idx2]["url"]))

            logger.info(f"predictions {predictions[0]}")

    df = pd.DataFrame(predictions, columns=["query", "language", "identifier", "url"])
    df.to_csv(training_ctx.output_dir / "predictions.csv", index=False)


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args["--debug"])
