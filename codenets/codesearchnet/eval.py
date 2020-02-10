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

from torch.utils.data import DataLoader

# from codenets.codesearchnet.single_branch_ctx import SingleBranchTrainingContext
from codenets.codesearchnet.dataset import (
    build_lang_dataset_single_code_tokenizer,
    BalancedBatchSchedulerSampler,
    DatasetType,
)
from codenets.codesearchnet.training_ctx import (
    CodeSearchTrainingContext,
    compute_loss_mrr,
    AvgLoss,
    AvgMrr,
    UsedTime,
    TotalLoss,
    TotalMrr,
    TotalSize,
    BatchSize,
    BatchLoss,
)


def run(args, tag_in_vcs=False) -> None:
    os.environ["WANDB_MODE"] = "dryrun"

    logger.debug("Building Training Context")
    training_ctx: CodeSearchTrainingContext
    restore_dir = args["--restore"]
    logger.info(f"Restoring Training Context from directory{restore_dir}")
    training_ctx = CodeSearchTrainingContext.build_context_from_dir(restore_dir)

    # Build Val Dataloader
    # val_dataset = training_ctx.build_lang_dataset(DatasetType.VAL)
    # val_dataloader = DataLoader(
    #     dataset=val_dataset,
    #     batch_size=training_ctx.val_batch_size,
    #     sampler=BalancedBatchSchedulerSampler(dataset=val_dataset, batch_size=training_ctx.val_batch_size),
    # )
    # logger.info(f"Built val_dataloader [Length:{len(val_dataloader)} x Batch:{training_ctx.val_batch_size}]")

    # Build Test Dataloader
    test_dataset = training_ctx.build_lang_dataset(DatasetType.TEST)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=training_ctx.val_batch_size,
        sampler=BalancedBatchSchedulerSampler(dataset=test_dataset, batch_size=training_ctx.test_batch_size),
    )
    logger.info(f"Built test_dataloader [Length:{len(test_dataloader)} x Batch:{training_ctx.test_batch_size}]")

    total_loss = TotalLoss(0.0)
    total_size = TotalSize(0)
    total_mrr = TotalMrr(0.0)
    training_ctx.eval_mode()
    with torch.no_grad():
        training_ctx.zero_grad()
    with tqdm(total=len(test_dataloader)) as t_batch:
        for batch_idx, batch in enumerate(test_dataloader):
            languages, similarity, query_tokens, query_tokens_mask, code_tokens, code_tokens_mask = [
                t.to(training_ctx.device) for t in batch
            ]

            per_sample_loss, per_sample_losses, similarity_scores = training_ctx.forward(batch, batch_idx)

            batch_size = BatchSize(batch[0].size()[0])
            batch_loss = BatchLoss(per_sample_loss.item())
            total_loss, avg_loss, total_mrr, avg_mrr, total_size = compute_loss_mrr(
                similarity_scores, batch_loss, batch_size, total_loss, total_mrr, total_size
            )
            #     languages=languages,
            #     query_tokens=query_tokens,
            #     query_tokens_mask=query_tokens_mask,
            #     code_tokens=code_tokens,
            #     code_tokens_mask=code_tokens_mask,
            # )
            # per_sample_losses, similarity_scores = training_ctx.losses_scores_fn(
            #     query_embedding, code_embedding, similarity
            # )
            # per_sample_loss = torch.mean(per_sample_losses)

            # nb_samples = batch[0].size()[0]

            # # compute MRR
            # # extract the logits from the diagonal of the matrix, which are the logits corresponding to the ground-truth
            # correct_scores = similarity_scores.diagonal()
            # # compute how many queries have bigger logits than the ground truth (the diagonal)
            # # the elements that are incorrectly ranked
            # compared_scores = similarity_scores.ge(correct_scores.unsqueeze(dim=-1)).float()
            # compared_scores_nb = torch.sum(compared_scores, dim=1)
            # per_sample_mrr = torch.div(1.0, compared_scores_nb)
            # per_batch_mrr = torch.sum(per_sample_mrr) / nb_samples

            # epoch_samples += nb_samples
            # epoch_loss += per_sample_loss.item() * nb_samples
            # loss = epoch_loss / max(1, epoch_samples)

            # mrr_sum += per_batch_mrr.item() * nb_samples
            # mrr = mrr_sum / max(1, epoch_samples)

            t_batch.set_postfix({f"loss": f"{per_sample_loss.item():10}"})
            t_batch.update(1)

    logger.info(
        f"total_loss:{total_loss}, avg_loss:{avg_loss}, total_mrr:{total_mrr}, avg_mrr:{avg_mrr}, total_size:{total_size}"
    )


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args["--debug"])
