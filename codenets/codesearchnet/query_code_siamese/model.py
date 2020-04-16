from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping, Optional, Union, Mapping, cast, Dict, List, Type

import numpy as np
import torch
from loguru import logger
from transformers import AdamW, BertConfig, BertModel, AlbertConfig, AlbertModel

from codenets.codesearchnet.data import DatasetParams
from codenets.codesearchnet.tokenizer_recs import TokenizerRecordable
from codenets.tensorboard_utils import Tensorboard
import wandb
from codenets.losses import load_loss_and_similarity_function
from codenets.codesearchnet.poolers import MeanWeightedPooler
from codenets.codesearchnet.huggingface.models import PreTrainedModelRecordable
from codenets.recordable import (
    HoconConfigRecordable,
    Recordable,
    RecordableMapping,
    RecordableTorchModule,
    RecordableTorchModuleMapping,
    DictRecordable,
    runtime_load_recordable_mapping,
    save_recordable_mapping,
)
from codenets.utils import full_classname, instance_full_classname
from pyhocon import ConfigTree


class QueryCodeSiamese(RecordableTorchModule):
    """
    A generic Pytorch Model with:
    - one single-branch query encoder
    - one single-branch code encoder
    - one optional pooler to pool output embeddings from any branch
    """

    def __init__(self, encoder: RecordableTorchModule, pooler: Optional[RecordableTorchModule] = None):
        super(QueryCodeSiamese, self).__init__()
        self.encoder = encoder
        self.pooler = pooler

    def save(self, output_dir: Union[Path, str]) -> bool:
        d = Path(output_dir) / instance_full_classname(self)
        records: MutableMapping[str, Recordable] = {"encoder": self.encoder}
        if self.pooler is not None:
            records["pooler"] = self.pooler
        return save_recordable_mapping(output_dir=d, records=records)

    @classmethod
    def load(cls, restore_dir: Union[Path, str]) -> QueryCodeSiamese:
        d = Path(restore_dir) / full_classname(cls)
        records = runtime_load_recordable_mapping(d)
        return cls(**records)

    def forward(  # type: ignore
        self,
        languages: np.ndarray,
        query_tokens: np.ndarray,
        query_tokens_mask: np.ndarray,
        code_tokens: np.ndarray,
        code_tokens_mask: np.ndarray,
        lang_weights: np.ndarray,
    ):
        # lang_id = str(languages[0].item())
        query_seq_outputs = self.encoder(query_tokens, query_tokens_mask)  # [B x S x H]
        code_seq_outputs = self.encoder(code_tokens, code_tokens_mask)  # [B x S x H]
        if self.pooler is not None:
            return (
                self.pooler(query_seq_outputs[0], query_tokens_mask),
                self.pooler(code_seq_outputs[0], code_tokens_mask),
            )
        else:
            # use already pooled data (need to be pretrained as it uses 1st (CLS) token logit)
            return query_seq_outputs[1], code_seq_outputs[1]

    def encode_query(self, query_tokens: np.ndarray, query_tokens_mask: np.ndarray) -> np.ndarray:
        query_seq_outputs = self.encoder(query_tokens, query_tokens_mask)

        if self.pooler is not None:
            return self.pooler(query_seq_outputs[0], query_tokens_mask)
        else:
            return query_seq_outputs[1]

    def encode_code(self, lang_id: int, code_tokens: np.ndarray, code_tokens_mask: np.ndarray) -> np.ndarray:
        code_seq_outputs = self.encoder(code_tokens, code_tokens_mask)
        if self.pooler is not None:
            return self.pooler(code_seq_outputs[0], code_tokens_mask)
        else:
            return code_seq_outputs[1]

    def tokenize_code(self, lang_id: int, code_tokens: np.ndarray, code_tokens_mask: np.ndarray) -> np.ndarray:
        code_seq_outputs = self.encoder(code_tokens, code_tokens_mask)
        if self.pooler is not None:
            return self.pooler(code_seq_outputs[0], code_tokens_mask)
        else:
            return code_seq_outputs[1]

    @classmethod
    def from_hocon(cls: Type[QueryCodeSiamese], config: ConfigTree) -> QueryCodeSiamese:
        """Load Query1Code1_CodeSearchModel from a config tree"""

        if "training.model.encoder.type" in config:
            if config["training.model.encoder.type"] == "albert":
                logger.info("Creating QueryCodeSiamese with Albert encoder")
                albert_config = AlbertConfig(**config["training.model.encoder"])
                encoder = PreTrainedModelRecordable(AlbertModel(albert_config))
            elif config["training.model.encoder.type"] == "bert":
                logger.info("Creating QueryCodeSiamese with Bert encoder")
                bert_config = BertConfig(**config["training.model.encoder"])
                encoder = PreTrainedModelRecordable(BertModel(bert_config))
        else:
            # default is BERT now
            logger.info("Creating QueryCodeSiamese with Bert encoder")
            bert_config = BertConfig(**config["training.model.encoder"])
            encoder = PreTrainedModelRecordable(BertModel(bert_config))

        model = QueryCodeSiamese(
            encoder=encoder, pooler=MeanWeightedPooler(input_size=config["training.model.encoder.hidden_size"])
        )

        return model
