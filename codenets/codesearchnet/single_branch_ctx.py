from typing import Mapping, cast, Type, List, Tuple, Iterable, Optional

# from pathlib import Path
# from loguru import logger
import torch
from torch import Tensor

# from torch import nn
import numpy as np
from transformers import AdamW
from pyhocon import ConfigTree

from codenets.recordable import Recordable
from codenets.codesearchnet.dataset import LangDataset, build_lang_dataset_single_code_tokenizer
from codenets.codesearchnet.training_ctx import CodeSearchTrainingContext, DatasetType
from codenets.codesearchnet.tokenizer_recs import TokenizerRecordable
from codenets.codesearchnet.training_ctx import ModelAndAdamWRecordable
from codenets.codesearchnet.single_branch_model import SingleBranchCodeSearchModel
from codenets.codesearchnet.tokenizer_recs import load_query_code_tokenizers_from_hocon_single_code_tokenizer


class SingleBranchCodeSearchModelAndAdamW(ModelAndAdamWRecordable):
    """
    Recordable for SingleBranchCodeSearchModelAndAdamW + Optimizer due to the fact
    that the optimizer can't be recovered without its model params... so
    we need to load both together or it's no generic.
    It is linked to optimizer AdamW because it's impossible to load
    something you don't know the class... Not very elegant too...
    For now, it doesn't manage the device of the model which is an issue
    but I haven't found an elegant solution to do that...
    To be continued
    """

    model_type = SingleBranchCodeSearchModel

    def __init__(self, model: SingleBranchCodeSearchModel, optimizer: AdamW):
        super(SingleBranchCodeSearchModelAndAdamW, self).__init__(model, optimizer)


class SingleBranchTrainingContext(CodeSearchTrainingContext):
    # SingleBranchCodeSearchModelAndAdamW = ModelAndAdamWRecordable[SingleBranchCodeSearchModel]

    def __init__(self, records: Mapping[str, Recordable]):
        super(SingleBranchTrainingContext, self).__init__(records)

        self.query_tokenizer = cast(TokenizerRecordable, records["query_tokenizer"])
        self.code_tokenizer = cast(TokenizerRecordable, records["code_tokenizer"])

        model_optimizer_rec = cast(SingleBranchCodeSearchModelAndAdamW, records["model_optimizer"])
        self.model = model_optimizer_rec.model
        self.model = self.model.to(device=self.device)
        self.optimizer = model_optimizer_rec.optimizer
        # trick to pass params to refresh params on the right device
        # as there is no function for that in pytorch
        # need to find a nicer solution...
        self.optimizer.load_state_dict(self.optimizer.state_dict())

    @classmethod
    def from_hocon_custom(cls: Type["SingleBranchTrainingContext"], conf: ConfigTree) -> Mapping[str, Recordable]:
        res = load_query_code_tokenizers_from_hocon_single_code_tokenizer(conf)
        if res is not None:
            query_tokenizer, code_tokenizer = res
        else:
            raise ValueError("Couldn't load Tokenizers from conf")

        device = torch.device(conf["training.device"])
        model = SingleBranchCodeSearchModel.from_hocon(conf)
        model = model.to(device=device)

        optimizer = AdamW(model.parameters(), lr=conf["training.lr"], correct_bias=False)

        records = {
            # need to pair model and optimizer as optimizer need it to be reloaded
            "model_optimizer": SingleBranchCodeSearchModelAndAdamW(model, optimizer),
            "query_tokenizer": query_tokenizer,
            "code_tokenizer": code_tokenizer,
        }

        return records

    def train_mode(self) -> bool:
        """Set all necessary elements in train mode"""
        self.model.train()
        torch.set_grad_enabled(True)
        return True

    def eval_mode(self) -> bool:
        """Set all necessary elements in train mode"""
        self.model.eval()
        torch.set_grad_enabled(False)
        return True

    def forward(self, batch: List[Tensor], batch_idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Perform forward path on batch
        
        Args:
            batch (List[Tensor]): the batch data as a List of tensors
            batch_idx (int): the batch index in dataloader
        
        Returns:
            Tuple[Tensor, Tensor, Tensor]: (global loss tensor for all samples in batch, losses per sample in batch, tensor matrix of similarity scores between all samples)
        """
        languages, similarity, query_tokens, query_tokens_mask, code_tokens, code_tokens_mask = [
            t.to(self.device) for t in batch
        ]
        (query_embedding, code_embedding) = self.model(
            languages=languages,
            query_tokens=query_tokens,
            query_tokens_mask=query_tokens_mask,
            code_tokens=code_tokens,
            code_tokens_mask=code_tokens_mask,
        )
        per_sample_losses, similarity_scores = self.losses_scores_fn(query_embedding, code_embedding, similarity)
        avg_loss = torch.mean(per_sample_losses)

        return (avg_loss, per_sample_losses, similarity_scores)

    def backward_optimize(self, loss: Tensor) -> Tensor:
        """Perform backward pass from loss"""
        loss.backward()
        self.optimizer.step()
        return loss

    def zero_grad(self) -> bool:
        """Set all necessary elements to zero_grad"""
        self.model.zero_grad()
        return True

    def encode_query(self, query_tokens: np.ndarray, query_tokens_mask: np.ndarray) -> np.ndarray:
        return self.model.encode_query(query_tokens, query_tokens_mask)

    def encode_code(self, lang_id: int, code_tokens, code_tokens_mask: np.ndarray) -> np.ndarray:
        return self.model.encode_code(lang_id, code_tokens, code_tokens_mask)

    def tokenize_query_sentences(
        self, sentences: List[str], max_length: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        return self.query_tokenizer.encode_sentences(sentences, max_length)

    def tokenize_code_sentences(
        self, sentences: List[str], max_length: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        return self.code_tokenizer.encode_sentences(sentences, max_length)

    def tokenize_code_tokens(
        self, tokens: Iterable[List[str]], max_length: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        return self.code_tokenizer.encode_tokens(tokens, max_length)

    def build_lang_dataset(self, dataset_type: DatasetType) -> LangDataset:
        """Build language dataset using custom training context tokenizers"""
        if dataset_type == DatasetType.TRAIN:
            return build_lang_dataset_single_code_tokenizer(
                self.train_dirs,
                f"train_{self.training_tokenizer_type}",
                self.train_data_params,
                self.query_tokenizer,
                self.code_tokenizer,
                lang_token="<lg>",
                pickle_path=self.pickle_path,
                parallelize=self.train_data_params.parallelize,
            )
        if dataset_type == DatasetType.VAL:
            return build_lang_dataset_single_code_tokenizer(
                self.val_dirs,
                f"val_{self.training_tokenizer_type}",
                self.val_data_params,
                self.query_tokenizer,
                self.code_tokenizer,
                lang_token="<lg>",
                pickle_path=self.pickle_path,
                parallelize=self.val_data_params.parallelize,
            )
        if dataset_type == DatasetType.TEST:
            return build_lang_dataset_single_code_tokenizer(
                self.test_dirs,
                f"test_{self.training_tokenizer_type}",
                self.test_data_params,
                self.query_tokenizer,
                self.code_tokenizer,
                lang_token="<lg>",
                pickle_path=self.pickle_path,
                parallelize=self.test_data_params.parallelize,
            )
        else:
            raise ValueError("DatasetType can only be TRAIN/VAL/TEST")
