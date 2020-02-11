from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping, Optional, Union, Mapping, cast, Dict, List

import numpy as np
import torch
from loguru import logger
from transformers import AdamW, BertConfig, BertModel

from codenets.codesearchnet.data import DatasetParams
from codenets.codesearchnet.tokenizer_recs import TokenizerRecordable, load_query_code_tokenizers_from_hocon
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
from codenets.utils import full_classname, instance_full_classname, expand_data_path
from pyhocon import ConfigTree


class Query1CodeN(RecordableTorchModule):
    """
    A generic Pytorch Model with:
    - one single-branch query encoder
    - multi-branch code encoders: one encoder per language
    - one optional pooler to pool output embeddings from any branch
    """

    def __init__(
        self,
        query_encoder: RecordableTorchModule,
        code_encoders: RecordableTorchModuleMapping,
        pooler: Optional[RecordableTorchModule] = None,
    ):
        super(Query1CodeN, self).__init__()
        self.code_encoders = code_encoders
        self.query_encoder = query_encoder
        self.pooler = pooler

    def save(self, output_dir: Union[Path, str]) -> bool:
        d = Path(output_dir) / instance_full_classname(self)
        records: MutableMapping[str, Recordable] = {
            "query_encoder": self.query_encoder,
            "code_encoders": self.code_encoders,
        }
        if self.pooler is not None:
            records["pooler"] = self.pooler
        return save_recordable_mapping(output_dir=d, records=records)

    @classmethod
    def load(cls, restore_dir: Union[Path, str]) -> Query1CodeN:
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
    ):
        lang_id = str(languages[0].item())
        query_seq_outputs = self.query_encoder(query_tokens, query_tokens_mask)  # [B x S x H]
        code_seq_outputs = self.code_encoders[lang_id](code_tokens, code_tokens_mask)  # [B x S x H]

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

        if self.pooler:
            return self.pooler(query_seq_outputs[0], query_tokens_mask)
        else:
            return query_seq_outputs[1]

    def encode_code(self, lang_id: int, code_tokens, code_tokens_mask: np.ndarray) -> np.ndarray:
        code_seq_outputs = self.code_encoders[str(lang_id)](code_tokens, code_tokens_mask)
        if self.pooler:
            return self.pooler(code_seq_outputs[0], code_tokens_mask)
        else:
            return code_seq_outputs[1]


def multibranch_bert_from_hocon(config: ConfigTree) -> Query1CodeN:
    """Load Query1CodeN from a config tree"""

    lang_ids = config["lang_ids"]
    query_config = BertConfig(**config["bert"])
    query_encoder = PreTrainedModelRecordable(BertModel(query_config))

    code_encoders: MutableMapping[str, RecordableTorchModule] = {}
    for language in lang_ids.keys():
        cfg = BertConfig(**config["bert"])
        module = PreTrainedModelRecordable(BertModel(cfg))
        lang_idx = lang_ids[language]
        code_encoders[str(lang_idx)] = module

    model = Query1CodeN(
        query_encoder=query_encoder,
        code_encoders=RecordableTorchModuleMapping(code_encoders),
        pooler=MeanWeightedPooler(input_size=query_config.hidden_size),
    )

    return model


class Query1CodeNAndAdamW(Recordable):
    """
    Recordable for Query1CodeN + Optimizer due to the fact
    that the optimizer can't be recovered without its model params... so 
    we need to load both together or it's no generic.
    It is linked to optimizer AdamW because it's impossible to load
    something you don't know the class... Not very elegant too...
    For now, it doesn't manage the device of the model which is an issue
    but I haven't found an elegant solution to do that...
    To be continued
    """

    def __init__(self, model: Query1CodeN, optimizer: AdamW):  # type: ignore
        super().__init__()
        self.model = model
        self.optimizer = optimizer

    def save(self, output_dir: Union[Path, str]) -> bool:
        full_dir = Path(output_dir) / instance_full_classname(self)
        logger.debug(f"Saving Query1CodeN & AdamW optimizer to {full_dir}")
        os.makedirs(full_dir, exist_ok=True)
        self.model.save(full_dir)
        torch.save(self.optimizer.state_dict(), full_dir / "adamw_state_dict.pth")
        return True

    @classmethod
    def load(cls, restore_dir: Union[Path, str]) -> Query1CodeNAndAdamW:
        full_dir = Path(restore_dir) / full_classname(cls)
        logger.debug(f"Loading Query1CodeN & AdamW optimizer from {full_dir}")
        model = Query1CodeN.load(full_dir)

        state_dict = torch.load(full_dir / "adamw_state_dict.pth")
        optimizer = AdamW(model.parameters())
        optimizer.load_state_dict(state_dict)
        return Query1CodeNAndAdamW(model, optimizer)


class MultiBranchTrainingContext(RecordableMapping):
    def __init__(self, records: Mapping[str, Recordable]):
        super(MultiBranchTrainingContext, self).__init__(records)

        # we need to cast elements back to their type when reading from records
        # so that mypy can help us again... from outside our world to inside of it
        self.conf = cast(HoconConfigRecordable, records["config"]).config

        self.training_name = self.conf["training.name"]
        self.training_iteration = self.conf["training.iteration"]
        self.training_full_name = f"{self.training_name}_{self.training_iteration}"

        self.device = torch.device(self.conf["training.device"])
        self.epochs = self.conf["training.epochs"]
        self.min_log_interval = self.conf["training.min_log_interval"]
        self.max_grad_norm = self.conf["training.max_grad_norm"]

        self.pickle_path = Path(self.conf["training.pickle_path"])
        self.tensorboard_path = Path(self.conf["training.tensorboard_path"])
        self.output_dir = Path(self.conf["training.output_dir"])

        self.train_data_params = DatasetParams(**self.conf["dataset.train.params"])
        self.train_dirs: List[Path] = expand_data_path(self.conf["dataset.train.dirs"])
        self.train_batch_size: int = self.conf["training.batch_size.train"]

        self.val_data_params = DatasetParams(**self.conf["dataset.val.params"])
        self.val_dirs: List[Path] = expand_data_path(self.conf["dataset.val.dirs"])
        self.val_batch_size: int = self.conf["training.batch_size.val"]

        self.test_data_params = DatasetParams(**self.conf["dataset.test.params"])
        self.test_dirs: List[Path] = expand_data_path(self.conf["dataset.test.dirs"])
        self.test_batch_size: int = self.conf["training.batch_size.test"]

        self.queries_file = self.conf["dataset.queries_file"]

        self.query_tokenizer = cast(TokenizerRecordable, records["query_tokenizer"])
        self.code_tokenizers = cast(Dict[str, TokenizerRecordable], records["code_tokenizers"])

        model_optimizer_rec = cast(Query1CodeNAndAdamW, records["model_optimizer"])
        self.model = model_optimizer_rec.model
        self.model = self.model.to(device=self.device)
        self.optimizer = model_optimizer_rec.optimizer
        # trick to pass params to refresh params on the right device
        # as there is no function for that in pytorch
        # need to find a nicer solution...
        self.optimizer.load_state_dict(self.optimizer.state_dict())

        self.losses_scores_fn = load_loss_and_similarity_function(self.conf["training.loss"], self.device)

        self.training_params = cast(DictRecordable, records["training_params"])
        self.epoch = self.training_params["epoch"]
        self.train_global_step = self.training_params["train_global_step"]
        self.val_global_step = self.training_params["val_global_step"]
        self.start_epoch = self.epoch
        self.best_epoch = self.start_epoch
        self.best_loss = self.training_params["val_loss"]
        self.best_mrr = self.training_params["val_mrr"]

        logger.info(
            f"Re-launching training from epoch: {self.start_epoch} with loss:{self.best_loss} mrr:{self.best_mrr}"
        )

    @classmethod
    def from_hocon(cls, conf: ConfigTree) -> MultiBranchTrainingContext:
        # train_data_params = DatasetParams(**conf["dataset.train.params"])
        # query_tokenizer, code_tokenizers = build_or_load_original_tokenizers(
        #     dirs=expand_data_path(conf["dataset.train.dirs"]),
        #     name="train",
        #     data_params=train_data_params,
        #     pickle_path=conf["training.pickle_path"],
        # )
        res = load_query_code_tokenizers_from_hocon(conf)
        if res is not None:
            query_tokenizer, code_tokenizers = res
        else:
            raise ValueError("Couldn't load Tokenizers from conf")

        device = torch.device(conf["training.device"])

        model = multibranch_bert_from_hocon(conf)
        model = model.to(device=device)

        optimizer = AdamW(model.parameters(), lr=conf["training.lr"], correct_bias=False)

        records = {
            "config": HoconConfigRecordable(conf),
            # need to pair model and optimizer as optimizer need it to be reloaded
            "model_optimizer": Query1CodeNAndAdamW(model, optimizer),
            "query_tokenizer": query_tokenizer,
            "code_tokenizers": code_tokenizers,
            "training_params": DictRecordable(
                {
                    "epoch": 0,
                    "train_global_step": 0,
                    "val_global_step": 0,
                    "train_loss": float("inf"),
                    "train_mrr": 0.0,
                    "val_loss": float("inf"),
                    "val_mrr": 0.0,
                }
            ),
        }
        return MultiBranchTrainingContext(records)
