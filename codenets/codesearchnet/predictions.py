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
    --restore DIR                    specify restoration dir.
    --wandb_run_id <entity_name>/<project_name>/<run_id> Specify Wandb Run
    --debug                          Enable debug routines. [default: False]
"""

import os
import sys
from pathlib import Path
from typing import Tuple
import torch
import numpy as np
from docopt import docopt
from dpu_utils.utils import run_and_debug
from loguru import logger
import pandas as pd
from annoy import AnnoyIndex
from tqdm import tqdm
import shutil
from wandb.apis import InternalApi
import wandb
from codenets.codesearchnet.training_ctx import CodeSearchTrainingContext


def compute_code_encodings_from_defs(
    language: str, training_ctx: CodeSearchTrainingContext, lang_token: str, batch_length: int = 1024
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f"Computing Encoding for language: {language}")
    lang_id = training_ctx.train_data_params.lang_ids[language]
    h5_file = (
        training_ctx.pickle_path
        / f"{language}_{training_ctx.training_full_name}_dedupe_definitions_v2_codes_encoded.h5"
    )
    root_data_path = Path(training_ctx.conf["dataset.root_dir"])

    def_file = root_data_path / f"data/{language}_dedupe_definitions_v2.pkl"
    definitions_df = pd.DataFrame(pd.read_pickle(open(def_file, "rb"), compression=None))
    cols_to_remove = list(definitions_df.columns.difference(["function_tokens", "identifier", "url"]))
    for col in cols_to_remove:
        del definitions_df[col]
    # definitions_df.drop(cols_to_remove, inplace=True, axis=1)
    logger.debug(f"definitions_df {definitions_df.columns}")

    if not os.path.exists(h5_file):
        logger.info(f"Building encodings of code from {def_file}")

        # function_tokens = definitions_df["function_tokens"]
        # add language and lang_token (<lg>) to tokens
        definitions_df["function_tokens"] = definitions_df["function_tokens"].apply(
            lambda row: [language, lang_token] + row
        )
        function_tokens_batch = definitions_df["function_tokens"].groupby(
            np.arange(len(definitions_df["function_tokens"])) // batch_length
        )

        code_embeddings = []
        for g, df_batch in tqdm(function_tokens_batch):
            # logger.debug(f"df_batch {df_batch.values}")
            codes_encoded, codes_masks = training_ctx.tokenize_code_tokens(
                df_batch.values, max_length=training_ctx.conf["dataset.common_params.code_max_num_tokens"]
            )

            # codes_encoded_t = torch.tensor(codes_encoded, dtype=torch.long).to(training_ctx.device)
            # codes_masks_t = torch.tensor(codes_masks, dtype=torch.long).to(training_ctx.device)

            # logger.debug(f"codes_encoded_t {codes_encoded_t}")
            # logger.debug(f"codes_masks_t {codes_masks_t}")

            emb_df = pd.DataFrame(
                training_ctx.encode_code(
                    lang_id=lang_id,
                    code_tokens=codes_encoded,
                    code_tokens_mask=codes_masks
                )
                # .cpu()
                # .numpy()
            )
            # logger.debug(f"codes_encoded_t:{codes_encoded_t.shape} codes_masks_t:{codes_masks_t.shape}")
            if g < 2:
                logger.debug(f"emb_df {emb_df.head()}")
            code_embeddings.append(emb_df)

        # free memory or it explodes on 32GB...
        del definitions_df["function_tokens"]

        code_embeddings_df = pd.concat(code_embeddings)

        logger.debug(f"code_embeddings_df {code_embeddings_df.head(20)}")

        code_embeddings_df.to_hdf(h5_file, key="code_embeddings_df", mode="w")
        return (code_embeddings_df, definitions_df)
    else:
        code_embeddings_df = pd.read_hdf(h5_file, key="code_embeddings_df")
        return (code_embeddings_df, definitions_df)


def run(args, tag_in_vcs=False) -> None:
    args_wandb_run_id = args["--wandb_run_id"]
    if args_wandb_run_id is not None:
        entity, project, name = args_wandb_run_id.split("/")
        os.environ["WANDB_RUN_ID"] = name
        os.environ["WANDB_RESUME"] = "must"

        wandb_api = wandb.Api()
        # retrieve saved model from W&B for this run
        logger.info("Fetching run from W&B...")
        try:
            wandb_api.run(args_wandb_run_id)
        except wandb.CommError:
            logger.error(f"ERROR: Problem querying W&B for wandb_run_id: {args_wandb_run_id}", file=sys.stderr)
            sys.exit(1)

    else:
        os.environ["WANDB_MODE"] = "dryrun"

    logger.debug("Building Training Context")
    training_ctx: CodeSearchTrainingContext
    restore_dir = args["--restore"]
    logger.info(f"Restoring Training Context from directory{restore_dir}")
    training_ctx = CodeSearchTrainingContext.build_context_from_dir(restore_dir)

    queries = pd.read_csv(training_ctx.queries_file)
    queries = list(map(lambda q: f"<qy> {q}", queries["query"].values))
    queries_tokens, queries_masks = training_ctx.tokenize_query_sentences(
        queries, max_length=training_ctx.conf["dataset.common_params.query_max_num_tokens"]
    )
    logger.info(f"queries: {queries}")

    training_ctx.eval_mode()
    with torch.no_grad():
        query_embeddings = (
            training_ctx.encode_query(
                query_tokens=queries_tokens,
                query_tokens_mask=queries_masks,
            )
            # .cpu()
            # .numpy()
        )
        logger.info(f"query_embeddings: {query_embeddings.shape}")

        topk = 100
        language_token = "<lg>"
        for lang_idx, language in enumerate(
            ("python", "go", "javascript", "java", "php", "ruby")
            # ("php", "ruby")
        ):  # in enumerate(("python", "go", "javascript", "java", "php", "ruby")):
            predictions = []
            # (codes_encoded_df, codes_masks_df, definitions) = get_language_defs(language, training_ctx, language_token)

            code_embeddings, definitions = compute_code_encodings_from_defs(
                language, training_ctx, language_token, batch_length=512
            )
            logger.info(f"Building Annoy Index of length {len(code_embeddings.values[0])}")
            indices: AnnoyIndex = AnnoyIndex(len(code_embeddings.values[0]), "angular")
            # idx = 0
            for index, emb in enumerate(tqdm(code_embeddings.values)):
                indices.add_item(index, emb)
            indices.build(10)

            for i, (query, query_embedding) in enumerate(tqdm(zip(queries, query_embeddings))):
                idxs, distances = indices.get_nns_by_vector(query_embedding, topk, include_distances=True)
                for idx2, _ in zip(idxs, distances):
                    predictions.append(
                        (query, language, definitions.iloc[idx2]["identifier"], definitions.iloc[idx2]["url"])
                    )

            logger.info(f"predictions {predictions[0]}")

            df = pd.DataFrame(predictions, columns=["query", "language", "identifier", "url"])
            # BUT WHY DOESNT IT WORK AS EXPECTED????
            df["query"] = df["query"].str.replace("<qy> ", "")
            df["identifier"] = df["identifier"].str.replace(",", "")
            df["identifier"] = df["identifier"].str.replace('"', "")
            df["identifier"] = df["identifier"].str.replace(";", "")
            df.to_csv(
                training_ctx.output_dir / f"model_predictions_{training_ctx.training_tokenizer_type}.csv",
                index=False,
                header=True if lang_idx == 0 else False,
                # mode="w" if lang_idx == 0 else "a",
                mode="a",
            )
            # Free memory
            del code_embeddings
            del definitions
            del predictions

    if args_wandb_run_id is not None:
        logger.info("Uploading predictions to W&B")
        # upload model predictions CSV file to W&B

        entity, project, name = args_wandb_run_id.split("/")

        # make sure the file is in our cwd, with the correct name
        predictions_csv = training_ctx.output_dir / f"model_predictions_{training_ctx.training_tokenizer_type}.csv"
        predictions_base_csv = "model_predictions.csv"
        shutil.copyfile(predictions_csv, predictions_base_csv)

        # Using internal wandb API. TODO: Update when available as a public API
        internal_api = InternalApi()
        internal_api.push([predictions_base_csv], run=name, entity=entity, project=project)


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args["--debug"])
