from __future__ import annotations

from pathlib import Path
from typing import MutableMapping, Optional, Union, Type

import numpy as np
from transformers import BertConfig, BertModel

from codenets.codesearchnet.poolers import MeanWeightedPooler
from codenets.codesearchnet.huggingface.models import PreTrainedModelRecordable
from codenets.recordable import (
    Recordable,
    RecordableTorchModule,
    runtime_load_recordable_mapping,
    save_recordable_mapping,
)
from codenets.utils import full_classname, instance_full_classname
from pyhocon import ConfigTree


class Query1Code1(RecordableTorchModule):
    """
    A generic Pytorch Model with:
    - one single-branch query encoder
    - one single-branch code encoder
    - one optional pooler to pool output embeddings from any branch
    """

    def __init__(
        self,
        query_encoder: RecordableTorchModule,
        code_encoder: RecordableTorchModule,
        pooler: Optional[RecordableTorchModule] = None,
    ):
        super(Query1Code1, self).__init__()
        self.code_encoder = code_encoder
        self.query_encoder = query_encoder
        self.pooler = pooler

    def save(self, output_dir: Union[Path, str]) -> bool:
        d = Path(output_dir) / instance_full_classname(self)
        records: MutableMapping[str, Recordable] = {
            "query_encoder": self.query_encoder,
            "code_encoder": self.code_encoder,
        }
        if self.pooler is not None:
            records["pooler"] = self.pooler
        return save_recordable_mapping(output_dir=d, records=records)

    @classmethod
    def load(cls, restore_dir: Union[Path, str]) -> Query1Code1:
        d = Path(restore_dir) / full_classname(cls)
        records = runtime_load_recordable_mapping(d)
        return cls(**records) # type:ignore[arg-type]

    def forward(
        self,
        languages: np.ndarray,
        query_tokens: np.ndarray,
        query_tokens_mask: np.ndarray,
        code_tokens: np.ndarray,
        code_tokens_mask: np.ndarray,
    ):
        # lang_id = str(languages[0].item())
        query_seq_outputs = self.query_encoder(query_tokens, query_tokens_mask)  # [B x S x H]
        code_seq_outputs = self.code_encoder(code_tokens, code_tokens_mask)  # [B x S x H]

        if self.pooler is not None:
            return (
                self.pooler(query_seq_outputs[0], query_tokens_mask),
                self.pooler(code_seq_outputs[0], code_tokens_mask),
            )
        else:
            # use already pooled data (need to be pretrained as it uses 1st (CLS) token logit)
            return query_seq_outputs[1], code_seq_outputs[1]

    def encode_query(self, query_tokens: np.ndarray, query_tokens_mask: np.ndarray) -> np.ndarray:
        query_seq_outputs = self.query_encoder(query_tokens, query_tokens_mask)

        if self.pooler is not None:
            return self.pooler(query_seq_outputs[0], query_tokens_mask)
        else:
            return query_seq_outputs[1]

    def encode_code(self, lang_id: int, code_tokens: np.ndarray, code_tokens_mask: np.ndarray) -> np.ndarray:
        code_seq_outputs = self.code_encoder(code_tokens, code_tokens_mask)
        if self.pooler is not None:
            return self.pooler(code_seq_outputs[0], code_tokens_mask)
        else:
            return code_seq_outputs[1]

    def tokenize_code(self, lang_id: int, code_tokens: np.ndarray, code_tokens_mask: np.ndarray) -> np.ndarray:
        code_seq_outputs = self.code_encoder(code_tokens, code_tokens_mask)
        if self.pooler is not None:
            return self.pooler(code_seq_outputs[0], code_tokens_mask)
        else:
            return code_seq_outputs[1]

    @classmethod
    def from_hocon(cls: Type[Query1Code1], config: ConfigTree) -> Query1Code1:
        """Load Query1Code1_CodeSearchModel from a config tree"""

        query_bert_config = BertConfig(**config["training.model.query_encoder"])
        query_encoder = PreTrainedModelRecordable(BertModel(query_bert_config))
        code_bert_config = BertConfig(**config["training.model.code_encoder"])
        code_encoder = PreTrainedModelRecordable(BertModel(code_bert_config))

        model = Query1Code1(
            query_encoder=query_encoder,
            code_encoder=code_encoder,
            pooler=MeanWeightedPooler(input_size=query_bert_config.hidden_size),
        )

        return model
