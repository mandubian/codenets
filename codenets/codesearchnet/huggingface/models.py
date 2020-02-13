# from __future__ import annotations

import os
from pathlib import Path
from typing import Union, TypeVar, Type, Generic
from loguru import logger
from transformers import BertModel, PreTrainedModel

from codenets.recordable import RecordableTorchModule
from codenets.utils import full_classname, instance_full_classname


PretrainedRec_T = TypeVar("PretrainedRec_T", bound="PreTrainedModelRecordable")
Pretrained_T = TypeVar("Pretrained_T", bound="PreTrainedModel")


class PreTrainedModelRecordable(Generic[Pretrained_T], RecordableTorchModule):
    """
    Wrap any generic HuggingFace PreTrainedModel as a Recordable Torch module
    equipped with load/save
    """

    def __init__(self, model: Pretrained_T):
        super().__init__()
        self.model = model

    def save(self, output_dir: Union[Path, str]) -> bool:
        full_dir = Path(output_dir) / instance_full_classname(self)
        logger.debug(f"Saving BertModel to {full_dir}")
        os.makedirs(full_dir, exist_ok=True)
        self.model.save_pretrained(full_dir)
        return True

    @classmethod
    def load(cls: Type[PretrainedRec_T], restore_dir: Union[Path, str]) -> PretrainedRec_T:
        full_dir = Path(restore_dir) / full_classname(cls)
        logger.debug(f"Loading BertModel from {full_dir}")
        model = BertModel.from_pretrained(str(full_dir))
        return cls(model)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)


# BertModelRecordable = PreTrainedModelRecordable[BertModel]


# class BertModelRecordable(RecordableTorchModule):
#     """
#     Wrapper to make BertModel recordable
#     Haven't found a way to make that generic in a typesafe mode,
#     mypy and generics are too limited but I'll search again because
#     all Transformers classes have the same save/load_pretrained functions
#     so in theory there is no reason no to have one single recordable for
#     them all
#     """

#     def __init__(self, model: BertModel):
#         super().__init__()
#         self.model = model

#     def save(self, output_dir: Union[Path, str]) -> bool:
#         full_dir = Path(output_dir) / instance_full_classname(self)
#         logger.debug(f"Saving BertModel to {full_dir}")
#         os.makedirs(full_dir, exist_ok=True)
#         self.model.save_pretrained(full_dir)
#         return True

#     @classmethod
#     def load(cls, restore_dir: Union[Path, str]) -> BertModelRecordable:
#         full_dir = Path(restore_dir) / full_classname(cls)
#         logger.debug(f"Loading BertModel from {full_dir}")
#         model = BertModel.from_pretrained(str(full_dir))
#         return BertModelRecordable(model)

#     def forward(self, tokens, tokens_mask):
#         return self.model.forward(tokens, tokens_mask)
