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
from typing import List, Tuple
import torch
import numpy as np
from docopt import docopt
from dpu_utils.utils import run_and_debug
from loguru import logger
import pandas as pd
from annoy import AnnoyIndex
import pickle
from tqdm import tqdm

from codenets.codesearchnet.single_branch_ctx import SingleBranchTrainingContext

# from codenets.utils import stream_load, stream_dump_elt


def get_language_defs(
    language: str,
    training_ctx: SingleBranchTrainingContext,
    lang_token: str
    # ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Evaluating language: %s" % language)
    h5_file = training_ctx.pickle_path / f"{language}_dedupe_definitions_v2_codes_encoded.h5"
    root_data_path = Path(training_ctx.conf["dataset.root_dir"])

    def_file = root_data_path / f"data/{language}_dedupe_definitions_v2.pkl"
    definitions = pd.DataFrame(pd.read_pickle(open(def_file, "rb"), compression=None))
    if not os.path.exists(h5_file):
        logger.info(f"Building encoding of code from {def_file}")

        # definitions = pickle.load(open(root_data_path / "data/{}_dedupe_definitions_v2.pkl".format(language), "rb"))
        # function_tokens = [[language, lang_token] + d["function_tokens"] for d in definitions]
        # pkl_fd = open(pkl_file, "wb")

        function_tokens = definitions["function_tokens"]
        function_tokens = function_tokens.apply(lambda row: [language, lang_token] + row)
        # function_tokens = function_tokens.apply(lambda )

        function_tokens_batch = function_tokens.groupby(np.arange(len(function_tokens)) // 128)

        codes_encoded_list = []
        codes_masks_list = []
        for g, df_b in tqdm(function_tokens_batch):
            codes_encoded, codes_masks = training_ctx.code_tokenizer.encode_tokens(df_b, max_length=200)
            codes_encoded_list.append(codes_encoded)
            codes_masks_list.append(codes_masks)

        codes_encoded_df = pd.DataFrame(codes_encoded)
        codes_masks_df = pd.DataFrame(codes_masks)

        codes_encoded_df.to_hdf(h5_file, key="codes_encoded_df", mode="w")
        codes_masks_df.to_hdf(h5_file, key="codes_masks_df", mode="a")
        return (codes_encoded_df, codes_masks_df, definitions)
    else:
        codes_encoded_df = pd.read_hdf(h5_file, key="codes_encoded_df")
        codes_masks_df = pd.read_hdf(h5_file, key="codes_masks_df")
        # definitions = pickle.load(open(root_data_path / "data/{}_dedupe_definitions_v2.pkl".format(language), "rb"))
        # (codes_encoded, codes_masks) = pickle.load(open(pkl_file, "rb"))

        return (codes_encoded_df, codes_masks_df, definitions)


def compute_code_embeddings(
    language: str,
    training_ctx: SingleBranchTrainingContext,
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
            # for x in range(0, 10, seg_length)
        ]

        code_embeddings = []
        for (codes_enc, codes_m) in tqdm(segments):
            logger.info(f"codes_enc {codes_enc}")
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


def compute_code_encodings_from_defs(
    language: str, training_ctx: SingleBranchTrainingContext, lang_token: str, batch_length: int = 1024
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f"Computing Encoding for language: {language}")
    lang_id = training_ctx.train_data_params.lang_ids[language]
    h5_file = training_ctx.pickle_path / f"{language}_dedupe_definitions_v2_codes_encoded.h5"
    root_data_path = Path(training_ctx.conf["dataset.root_dir"])

    def_file = root_data_path / f"data/{language}_dedupe_definitions_v2.pkl"
    definitions_df = pd.DataFrame(pd.read_pickle(open(def_file, "rb"), compression=None))
    if not os.path.exists(h5_file):
        logger.info(f"Building encodings of code from {def_file}")

        function_tokens = definitions_df["function_tokens"]
        # add language and lang_token (<lg>) to tokens
        function_tokens = function_tokens.apply(lambda row: [language, lang_token] + row)
        function_tokens_batch = function_tokens.groupby(np.arange(len(function_tokens)) // batch_length)

        code_embeddings = []
        for g, df_batch in tqdm(function_tokens_batch):
            # logger.debug(f"df_batch {df_batch.values}")
            codes_encoded, codes_masks = training_ctx.code_tokenizer.encode_tokens(
                df_batch.values, max_length=training_ctx.conf["dataset.common_params.code_max_num_tokens"]
            )

            codes_encoded_t = torch.tensor(codes_encoded, dtype=torch.long).to(training_ctx.device)
            codes_masks_t = torch.tensor(codes_masks, dtype=torch.long).to(training_ctx.device)

            # logger.debug(f"codes_encoded_t {codes_encoded_t}")
            # logger.debug(f"codes_masks_t {codes_masks_t}")

            emb_df = pd.DataFrame(
                training_ctx.model.encode_code(
                    lang_id=lang_id, code_tokens=codes_encoded_t, code_tokens_mask=codes_masks_t
                )
                .cpu()
                .numpy()
            )
            # logger.debug(f"codes_encoded_t:{codes_encoded_t.shape} codes_masks_t:{codes_masks_t.shape}")
            if g < 2:
                logger.debug(f"emb_df {emb_df.head()}")
            code_embeddings.append(emb_df)

        code_embeddings_df = pd.concat(code_embeddings)

        logger.debug(f"code_embeddings_df {code_embeddings_df.head(20)}")

        code_embeddings_df.to_hdf(h5_file, key="code_embeddings_df", mode="w")
        return (code_embeddings_df, definitions_df)
    else:
        code_embeddings_df = pd.read_hdf(h5_file, key="code_embeddings_df")
        return (code_embeddings_df, definitions_df)


def run(args, tag_in_vcs=False) -> None:
    os.environ["WANDB_MODE"] = "dryrun"

    logger.debug("Building Training Context")
    training_ctx: SingleBranchTrainingContext
    restore_dir = args["--restore"]
    logger.info(f"Restoring Training Context from directory{restore_dir}")
    training_ctx = SingleBranchTrainingContext.load(restore_dir)

    queries = pd.read_csv(training_ctx.queries_file)
    queries = list(queries["query"].values)
    queries_tokens, queries_masks = training_ctx.query_tokenizer.encode_sentences(
        queries, max_length=training_ctx.conf["dataset.common_params.query_max_num_tokens"]
    )
    logger.info(f"queries_tokens: {queries_tokens}")

    training_ctx.eval_mode()
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
        language_token = "<lg>"
        for language in ("python",):  # , "go", "javascript", "java", "php", "ruby"):
            # (codes_encoded_df, codes_masks_df, definitions) = get_language_defs(language, training_ctx, language_token)

            code_embeddings, definitions = compute_code_encodings_from_defs(language, training_ctx, language_token)
            logger.debug(f"definitions {definitions.iloc[0]}")
            logger.debug(f"code_embeddings {code_embeddings.values[:5]}")
            logger.info(f"Building Annoy Index of length {len(code_embeddings.values[0])}")
            # indices: AnnoyIndex = AnnoyIndex(code_embeddings[0][0].shape[0], "angular")
            indices: AnnoyIndex = AnnoyIndex(len(code_embeddings.values[0]), "angular")
            # idx = 0
            for index, emb in enumerate(tqdm(code_embeddings.values)):
                # logger.info(f"vectors {vectors}")
                # for vector in vectors:
                # if vector is not None:
                # if idx < 10:
                # logger.debug(f"vector {len(vector)}")
                # indices.add_item(idx, vector)
                # idx += 1
                indices.add_item(index, emb)
            indices.build(10)

            for i, (query, query_embedding) in enumerate(tqdm(zip(queries, query_embeddings))):
                idxs, distances = indices.get_nns_by_vector(query_embedding, topk, include_distances=True)
                if i < 5:
                    logger.debug(f"query_embedding {query_embedding}")
                    logger.debug(f"idxs:{idxs}, distances:{distances}")
                for idx2, _ in zip(idxs, distances):
                    predictions.append(
                        (query, language, definitions.iloc[idx2]["identifier"], definitions.iloc[idx2]["url"])
                    )

            logger.info(f"predictions {predictions[0]}")
            del code_embeddings
            del definitions

    df = pd.DataFrame(predictions, columns=["query", "language", "identifier", "url"])
    df.to_csv(training_ctx.output_dir / "predictions.csv", index=False)


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args["--debug"])
