from __future__ import annotations

import os
from pathlib import Path
from typing import Union, TypeVar, Type, Generic
from loguru import logger
from transformers import PreTrainedModel

from codenets.recordable import RecordableTorchModule
from codenets.utils import full_classname, instance_full_classname, runtime_import


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
        full_dir = Path(output_dir) / instance_full_classname(self) / instance_full_classname(self.model)
        logger.info(f"Saving HuggingFace model to {full_dir}")
        os.makedirs(full_dir, exist_ok=True)
        self.model.save_pretrained(full_dir)
        return True

    @classmethod
    def load(cls: Type[PretrainedRec_T], restore_dir: Union[Path, str]) -> PretrainedRec_T:
        full_dir = Path(restore_dir) / full_classname(cls)
        logger.info(f"Loading HuggingFace Pretrained model from {full_dir}")
        _, dirs, _ = list(os.walk(full_dir))[0]
        model_cls_name = dirs[0]
        logger.info(f"Loading HuggingFace {model_cls_name} model from {full_dir}/{model_cls_name}")
        klass = runtime_import(model_cls_name)
        assert issubclass(klass, PreTrainedModel)

        model = klass.from_pretrained(str(full_dir / model_cls_name))

        return cls(model)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)