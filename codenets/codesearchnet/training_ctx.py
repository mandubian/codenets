import os
import torch
from typing import Mapping, cast, Optional, List, TypeVar, Type, Tuple, Union, Generic, NewType, Iterable, Callable
from pathlib import Path
from loguru import logger
import wandb
from pyhocon import ConfigTree
from torch import Tensor
from transformers import AdamW
import numpy as np
from codenets.tensorboard_utils import Tensorboard
from codenets.recordable import (
    RecordableMapping,
    Recordable,
    HoconConfigRecordable,
    DictRecordable,
    runtime_load_recordable,
)
from codenets.codesearchnet.data import DatasetParams
from codenets.codesearchnet.dataset_utils import LangDataset, DatasetType
from codenets.utils import expand_data_path, instance_full_classname, full_classname, runtime_import
from codenets.losses import load_loss_and_similarity_function


def default_sample_update(tpe: str, lang: str, tokens: List[str]) -> str:
    return " ".join(tokens) + "\r\n"


Model_T = TypeVar("Model_T", bound="RecordableTorchModule")  # type: ignore
ModelAndAdamWRecordable_T = TypeVar("ModelAndAdamWRecordable_T", bound="ModelAndAdamWRecordable")

# Testing Newtypes to force more type safety in code
BatchLoss = NewType("BatchLoss", float)
BatchSize = NewType("BatchSize", int)
TotalLoss = NewType("TotalLoss", float)
TotalSize = NewType("TotalSize", int)
TotalMrr = NewType("TotalMrr", float)
AvgLoss = NewType("AvgLoss", float)
AvgMrr = NewType("AvgMrr", float)
UsedTime = NewType("UsedTime", float)


def compute_loss_mrr(
    similarity_scores: Tensor,
    batch_loss: BatchLoss,
    batch_size: BatchSize,
    total_loss: TotalLoss,
    total_mrr: TotalMrr,
    total_samples: TotalSize,
) -> Tuple[TotalLoss, AvgLoss, TotalMrr, AvgMrr, TotalSize]:
    # compute total loss, avg loss, total MRR, avg MRR, total size
    # extract the logits from the diagonal of the matrix, which are the logits corresponding to the ground-truth
    correct_scores = similarity_scores.diagonal()
    # compute how many queries have bigger logits than the ground truth (the diagonal)
    # the elements that are incorrectly ranked
    compared_scores = similarity_scores.ge(correct_scores.unsqueeze(dim=-1)).float()
    compared_scores_nb = torch.sum(compared_scores, dim=1)
    per_sample_mrr = torch.div(1.0, compared_scores_nb)
    per_batch_mrr = torch.sum(per_sample_mrr) / batch_size

    new_total_samples = TotalSize(total_samples + batch_size)
    new_total_loss = TotalLoss(total_loss + batch_loss * batch_size)
    avg_loss = AvgLoss(new_total_loss / max(1, new_total_samples))

    new_total_mrr = TotalMrr(total_mrr + per_batch_mrr.item() * batch_size)
    avg_mrr = AvgMrr(new_total_mrr / max(1, new_total_samples))
    return new_total_loss, avg_loss, new_total_mrr, avg_mrr, new_total_samples


class ModelAndAdamWRecordable(Generic[Model_T], Recordable):
    """
    Recordable for RecordableTorchModule + Optimizer due to the fact
    that the optimizer can't be recovered without its model params... so 
    we need to load both together or it's no generic.
    It is linked to optimizer AdamW because it's impossible to load
    something you don't know the class... Not very elegant too...
    For now, it doesn't manage the device of the model which is an issue
    but I haven't found an elegant solution to do that...
    To be continued
    """

    model_type: Model_T

    def __init__(self, model: Model_T, optimizer: AdamW):  # type: ignore
        super().__init__()
        self.model = model
        self.optimizer = optimizer

    def save(self, output_dir: Union[Path, str]) -> bool:
        full_dir = Path(output_dir) / instance_full_classname(self)
        logger.debug(f"Saving {instance_full_classname(self)} & AdamW optimizer to {full_dir}")
        os.makedirs(full_dir, exist_ok=True)
        self.model.save(full_dir)
        torch.save(self.optimizer.state_dict(), full_dir / "adamw_state_dict.pth")
        return True

    @classmethod
    def load(cls: Type[ModelAndAdamWRecordable_T], restore_dir: Union[Path, str]) -> "ModelAndAdamWRecordable_T":
        full_dir = Path(restore_dir) / full_classname(cls)
        logger.debug(f"Loading {full_classname(cls)} & AdamW optimizer from {full_dir}")
        model = cls.model_type.load(full_dir)

        state_dict = torch.load(full_dir / "adamw_state_dict.pth")
        optimizer = AdamW(model.parameters())
        optimizer.load_state_dict(state_dict)
        return cls(model, optimizer)


CodeSearchTrainingContext_T = TypeVar("CodeSearchTrainingContext_T", bound="CodeSearchTrainingContext")


class CodeSearchTrainingContext(RecordableMapping):
    def __init__(self, records: Mapping[str, Recordable]):
        super(CodeSearchTrainingContext, self).__init__(records)

        # we need to cast elements back to their type when reading from records
        # so that mypy can help us again... from outside our world to inside of it
        self.conf = cast(HoconConfigRecordable, records["config"]).config

        self.training_name = self.conf["training.name"]
        self.training_tokenizer_type = self.conf["training.tokenizer_type"]
        self.training_iteration = self.conf["training.iteration"]
        self.training_full_name = f"{self.training_name}_{self.training_iteration}"

        self.device = torch.device(self.conf["training.device"])
        self.epochs = self.conf["training.epochs"]
        self.min_log_interval = self.conf["training.min_log_interval"]
        self.max_grad_norm = self.conf["training.max_grad_norm"]

        self.pickle_path = Path(self.conf["training.pickle_path"])
        self.tensorboard_activated = self.conf["training.tensorboard"]
        self.tensorboard_path = Path(self.conf["training.tensorboard_path"])
        self.tensorboard: Optional[Tensorboard] = None
        if self.tensorboard_activated:
            logger.info("Activating Tensorboard")
            self.tensorboard = Tensorboard(
                output_dir=self.tensorboard_path, experiment_id=self.training_name, unique_id=self.training_iteration
            )

        self.wandb_activated = self.conf["training.wandb"]
        if not self.wandb_activated:
            logger.info("Deactivating WanDB")
            os.environ["WANDB_MODE"] = "dryrun"
        else:
            logger.info("Activating WanDB")
            wandb.init(name=self.training_full_name, config=self.conf.as_plain_ordered_dict())

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

        # self.query_tokenizer = cast(TokenizerRecordable, records["query_tokenizer"])
        # self.code_tokenizer = cast(TokenizerRecordable, records["code_tokenizer"])

        self.losses_scores_fn = load_loss_and_similarity_function(self.conf["training.loss"], self.device)

        self.training_params = cast(DictRecordable, records["training_params"])
        self.epoch = self.training_params["epoch"]
        self.train_global_step = self.training_params["train_global_step"]
        self.val_global_step = self.training_params["val_global_step"]
        self.start_epoch = self.epoch
        self.best_loss_epoch = self.start_epoch
        self.best_loss = self.training_params["val_loss"]
        self.best_mrr_epoch = self.start_epoch
        self.best_mrr = self.training_params["val_mrr"]

        logger.info(
            f"Re-launching training from epoch: {self.start_epoch} with loss:{self.best_loss} mrr:{self.best_mrr}"
        )

    @classmethod
    def from_hocon(cls: Type[CodeSearchTrainingContext_T], conf: ConfigTree) -> CodeSearchTrainingContext_T:
        records = {
            "config": HoconConfigRecordable(conf),
            # need to pair model and optimizer as optimizer need it to be reloaded
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

        records.update(cls.from_hocon_custom(conf))

        return cls(records)

    @classmethod
    def from_hocon_custom(cls: Type[CodeSearchTrainingContext_T], conf: ConfigTree) -> Mapping[str, Recordable]:
        """Add custom recordable elements at load... To be implemented in custom Training Ctx"""
        pass

    def train_mode(self) -> bool:
        """Set all necessary elements in train mode"""
        pass

    def eval_mode(self) -> bool:
        """Set all necessary elements in train mode"""
        pass

    def forward(self, batch: List[Tensor], batch_idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Perform forward path on batch
        
        Args:
            batch (List[Tensor]): the batch data as a List of tensors
            batch_idx (int): the batch index in dataloader
        
        Returns:
            Tuple[Tensor, Tensor, Tensor]: (global loss tensor for all samples in batch, losses per sample in batch, tensor matrix of similarity scores between all samples)
        """
        pass

    def backward_optimize(self, loss: Tensor) -> Tensor:
        """Perform backward pass from loss"""
        pass

    def zero_grad(self) -> bool:
        """Set all necessary elements to zero_grad"""
        pass

    def build_lang_dataset(self, dataset_type: DatasetType) -> LangDataset:
        """Build language dataset using custom training context tokenizers"""
        pass

    def encode_query(self, query_tokens: np.ndarray, query_tokens_mask: np.ndarray) -> np.ndarray:
        pass

    def encode_code(self, lang_id: int, code_tokens, code_tokens_mask: np.ndarray) -> np.ndarray:
        pass

    def tokenize_query_sentences(
        self, sentences: List[str], max_length: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        pass

    def tokenize_code_sentences(
        self, sentences: List[str], max_length: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        pass

    def tokenize_code_tokens(
        self, tokens: Iterable[List[str]], max_length: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        pass

    def build_tokenizers(
        self,
        from_dataset_type: DatasetType,
        sample_update: Callable[[str, str, List[str]], str] = default_sample_update,
    ) -> bool:
        pass

    @classmethod
    def build_context_from_hocon(cls, conf: ConfigTree) -> "CodeSearchTrainingContext":
        """
        Build Training Context from Hocon config field training.model.training_clx_class
        Class is loaded at runtime because we can't know it before reading the configuration.

        Returns:
            CodeSearchTrainingContext_T: A instance of a training context subclass of CodeSearchTrainingContext (or crashes)
        """
        klass = runtime_import(conf["training.model.training_ctx_class"])
        assert issubclass(klass, CodeSearchTrainingContext)
        return klass.from_hocon(conf)

    @classmethod
    def build_context_from_dir(cls, dir: Path) -> "CodeSearchTrainingContext":
        """
        Build Training Context from recorded directory
        Class is loaded at runtime because we can't know it before reading the configuration.

        Returns:
            CodeSearchTrainingContext_T: A instance of a training context subclass of CodeSearchTrainingContext (or crashes)
        """
        ctx = runtime_load_recordable(dir)
        assert isinstance(ctx, CodeSearchTrainingContext)
        return ctx
