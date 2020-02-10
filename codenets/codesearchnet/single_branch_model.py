from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping, Optional, Union, Mapping, cast, Dict, List, Type

import numpy as np
import torch
from loguru import logger
from transformers import AdamW, BertConfig, BertModel

from codenets.codesearchnet.data import DatasetParams
from codenets.codesearchnet.tokenizer_recs import (
    TokenizerRecordable,
    load_query_code_tokenizers_from_hocon_single_code_tokenizer,
)
from codenets.tensorboard_utils import Tensorboard
import wandb
from codenets.losses import load_loss_and_similarity_function
from codenets.codesearchnet.poolers import MeanWeightedPooler
from codenets.codesearchnet.huggingface import PreTrainedModelRecordable
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


class SingleBranchCodeSearchModel(RecordableTorchModule):
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
        super(SingleBranchCodeSearchModel, self).__init__()
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
    def load(cls, restore_dir: Union[Path, str]) -> SingleBranchCodeSearchModel:
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
    def from_hocon(cls: Type[SingleBranchCodeSearchModel], config: ConfigTree) -> SingleBranchCodeSearchModel:
        """Load SingleBranchCodeSearchModel from a config tree"""

        # lang_ids = config["lang_ids"]
        query_bert_config = BertConfig(**config["training.model.query_encoder"])
        query_encoder = PreTrainedModelRecordable(BertModel(query_bert_config))
        code_bert_config = BertConfig(**config["training.model.code_encoder"])
        code_encoder = PreTrainedModelRecordable(BertModel(code_bert_config))

        model = SingleBranchCodeSearchModel(
            query_encoder=query_encoder,
            code_encoder=code_encoder,
            pooler=MeanWeightedPooler(input_size=query_bert_config.hidden_size),
        )

        return model


# class SingleBranchCodeSearchModelAndAdamW(Recordable):
#     """
#     Recordable for MultiBranchCodeSearchModel + Optimizer due to the fact
#     that the optimizer can't be recovered without its model params... so
#     we need to load both together or it's no generic.
#     It is linked to optimizer AdamW because it's impossible to load
#     something you don't know the class... Not very elegant too...
#     For now, it doesn't manage the device of the model which is an issue
#     but I haven't found an elegant solution to do that...
#     To be continued
#     """

#     def __init__(self, model: SingleBranchCodeSearchModel, optimizer: AdamW):  # type: ignore
#         super().__init__()
#         self.model = model
#         self.optimizer = optimizer

#     def save(self, output_dir: Union[Path, str]) -> bool:
#         full_dir = Path(output_dir) / instance_full_classname(self)
#         logger.debug(f"Saving SingleBranchCodeSearchModel & AdamW optimizer to {full_dir}")
#         os.makedirs(full_dir, exist_ok=True)
#         self.model.save(full_dir)
#         torch.save(self.optimizer.state_dict(), full_dir / "adamw_state_dict.pth")
#         return True

#     @classmethod
#     def load(cls, restore_dir: Union[Path, str]) -> SingleBranchCodeSearchModelAndAdamW:
#         full_dir = Path(restore_dir) / full_classname(cls)
#         logger.debug(f"Loading SingleBranchCodeSearchModelAndAdamW & AdamW optimizer from {full_dir}")
#         model = SingleBranchCodeSearchModel.load(full_dir)

#         state_dict = torch.load(full_dir / "adamw_state_dict.pth")
#         optimizer = AdamW(model.parameters())
#         optimizer.load_state_dict(state_dict)
#         return SingleBranchCodeSearchModelAndAdamW(model, optimizer)


# class SingleBranchTrainingContext(RecordableMapping):
#     def __init__(self, records: Mapping[str, Recordable]):
#         super(SingleBranchTrainingContext, self).__init__(records)

#         # we need to cast elements back to their type when reading from records
#         # so that mypy can help us again... from outside our world to inside of it
#         self.conf = cast(HoconConfigRecordable, records["config"]).config

#         self.training_name = self.conf["training.name"]
#         self.training_iteration = self.conf["training.iteration"]
#         self.training_full_name = f"{self.training_name}_{self.training_iteration}"

#         self.device = torch.device(self.conf["training.device"])
#         self.epochs = self.conf["training.epochs"]
#         self.min_log_interval = self.conf["training.min_log_interval"]
#         self.max_grad_norm = self.conf["training.max_grad_norm"]

#         self.pickle_path = Path(self.conf["training.pickle_path"])
#         self.tensorboard_activated = self.conf["training.tensorboard"]
#         self.tensorboard_path = Path(self.conf["training.tensorboard_path"])
#         self.tensorboard: Optional[Tensorboard] = None
#         if self.tensorboard_activated:
#             logger.info("Activating Tensorboard")
#             self.tensorboard = Tensorboard(
#                 output_dir=self.tensorboard_path, experiment_id=self.training_name, unique_id=self.training_iteration
#             )
#             # cfg = self.conf.as_plain_ordered_dict()
#             # self.tensorboard.add_scalars(cfg, global_step=0, group="train")
#             # self.tensorboard.add_scalars(cfg, global_step=0, group="val")

#         self.wandb_activated = self.conf["training.wandb"]
#         if not self.wandb_activated:
#             logger.info("Deactivating WanDB")
#             os.environ["WANDB_MODE"] = "dryrun"
#         else:
#             logger.info("Activating WanDB")
#             wandb.init(name=self.training_full_name, config=self.conf.as_plain_ordered_dict())

#         self.output_dir = Path(self.conf["training.output_dir"])

#         self.train_data_params = DatasetParams(**self.conf["dataset.train.params"])
#         self.train_dirs: List[Path] = expand_data_path(self.conf["dataset.train.dirs"])
#         self.train_batch_size: int = self.conf["training.batch_size.train"]

#         self.val_data_params = DatasetParams(**self.conf["dataset.val.params"])
#         self.val_dirs: List[Path] = expand_data_path(self.conf["dataset.val.dirs"])
#         self.val_batch_size: int = self.conf["training.batch_size.val"]

#         self.test_data_params = DatasetParams(**self.conf["dataset.test.params"])
#         self.test_dirs: List[Path] = expand_data_path(self.conf["dataset.test.dirs"])
#         self.test_batch_size: int = self.conf["training.batch_size.test"]

#         self.queries_file = self.conf["dataset.queries_file"]

#         # self.query_tokenizer = cast(TokenizerRecordable, records["query_tokenizer"])
#         # self.code_tokenizer = cast(TokenizerRecordable, records["code_tokenizer"])

#         # model_optimizer_rec = cast(SingleBranchCodeSearchModelAndAdamW, records["model_optimizer"])
#         # self.model = model_optimizer_rec.model
#         # self.model = self.model.to(device=self.device)
#         # self.optimizer = model_optimizer_rec.optimizer
#         # # trick to pass params to refresh params on the right device
#         # # as there is no function for that in pytorch
#         # # need to find a nicer solution...
#         # self.optimizer.load_state_dict(self.optimizer.state_dict())

#         self.losses_scores_fn = load_loss_and_similarity_function(self.conf["training.loss"], self.device)

#         self.training_params = cast(DictRecordable, records["training_params"])
#         self.epoch = self.training_params["epoch"]
#         self.train_global_step = self.training_params["train_global_step"]
#         self.val_global_step = self.training_params["val_global_step"]
#         self.start_epoch = self.epoch
#         self.best_epoch = self.start_epoch
#         self.best_loss = self.training_params["val_loss"]
#         self.best_mrr = self.training_params["val_mrr"]

#         logger.info(
#             f"Re-launching training from epoch: {self.start_epoch} with loss:{self.best_loss} mrr:{self.best_mrr}"
#         )

#     @classmethod
#     def from_hocon(cls, conf: ConfigTree) -> SingleBranchTrainingContext:
#         res = load_query_code_tokenizers_from_hocon_single_code_tokenizer(conf)
#         if res is not None:
#             query_tokenizer, code_tokenizer = res
#         else:
#             raise ValueError("Couldn't load Tokenizers from conf")

#         device = torch.device(conf["training.device"])

#         model = singlebranch_bert_from_hocon(conf)
#         model = model.to(device=device)

#         optimizer = AdamW(model.parameters(), lr=conf["training.lr"], correct_bias=False)

#         records = {
#             "config": HoconConfigRecordable(conf),
#             # need to pair model and optimizer as optimizer need it to be reloaded
#             "model_optimizer": SingleBranchCodeSearchModelAndAdamW(model, optimizer),
#             "query_tokenizer": query_tokenizer,
#             "code_tokenizer": code_tokenizer,
#             "training_params": DictRecordable(
#                 {
#                     "epoch": 0,
#                     "train_global_step": 0,
#                     "val_global_step": 0,
#                     "train_loss": float("inf"),
#                     "train_mrr": 0.0,
#                     "val_loss": float("inf"),
#                     "val_mrr": 0.0,
#                 }
#             ),
#         }
#         return SingleBranchTrainingContext(records)
