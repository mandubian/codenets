from typing import Mapping, cast, Type, List, Tuple, Iterable, Optional, Callable, Union, Dict

import os
import torch
from torch import Tensor
from loguru import logger
from pathlib import Path
import time

# from torch import nn
import numpy as np
from transformers import AdamW
from pyhocon import ConfigTree
from tokenizers import BPETokenizer
from tokenizers.normalizers import BertNormalizer
from tokenizers.trainers import BpeTrainer
from codenets.recordable import Recordable, RecordableMapping, NoneRecordable, DictRecordable
from codenets.codesearchnet.dataset_utils import LangDataset
from codenets.codesearchnet.data import DatasetParams
from codenets.codesearchnet.training_ctx import CodeSearchTrainingContext, DatasetType
from codenets.codesearchnet.tokenizer_recs import TokenizerRecordable
from codenets.codesearchnet.training_ctx import ModelAndAdamWRecordable
from codenets.utils import _to_subtoken_stream

from codenets.codesearchnet.query_code_siamese.model import QueryCodeSiamese
from codenets.codesearchnet.query_code_siamese.dataset import build_lang_dataset_siamese_tokenizer
from codenets.codesearchnet.huggingface.tokenizer_recs import (
    HuggingfaceBPETokenizerRecordable,
    build_huggingface_token_files,
)


class QueryCodeSiameseModelAndAdamW(ModelAndAdamWRecordable):
    """
    Recordable for Query1Code1ModelAndAdamW + Optimizer due to the fact
    that the optimizer can't be recovered without its model params... so
    we need to load both together or it's no generic.
    It is linked to optimizer AdamW because it's impossible to load
    something you don't know the class... Not very elegant too...
    For now, it doesn't manage the device of the model which is an issue
    but I haven't found an elegant solution to do that...
    To be continued
    """

    model_type = QueryCodeSiamese

    def __init__(self, model: QueryCodeSiamese, optimizer: AdamW):
        super(QueryCodeSiameseModelAndAdamW, self).__init__(model, optimizer)


class QueryCodeSiameseCtx(CodeSearchTrainingContext):
    def __init__(self, records: Mapping[str, Recordable]):
        super(QueryCodeSiameseCtx, self).__init__(records)
        logger.info("Loading QueryCodeSiameseCtx")
        # TODO manage the NoneRecordable case or not?
        self.tokenizer = cast(TokenizerRecordable, records["tokenizer"])
        self.common_tokens: Optional[DictRecordable]
        if "common_tokens" in records:
            self.common_tokens = cast(DictRecordable, records["common_tokens"])
        else:
            self.common_tokens = None

        model_optimizer_rec = cast(QueryCodeSiameseModelAndAdamW, records["model_optimizer"])
        self.model = model_optimizer_rec.model
        self.model = self.model.to(device=self.device)
        self.optimizer = model_optimizer_rec.optimizer
        # trick to pass params to refresh params on the right device
        # as there is no function for that in pytorch
        # need to find a nicer solution...
        self.optimizer.load_state_dict(self.optimizer.state_dict())

    @classmethod
    def from_hocon_custom(cls: Type["QueryCodeSiameseCtx"], conf: ConfigTree) -> Mapping[str, Recordable]:
        tokenizer = load_tokenizers_from_hocon(conf)
        common_tokens = load_common_tokens_from_hocon(conf)

        device = torch.device(conf["training.device"])
        model = QueryCodeSiamese.from_hocon(conf)
        model = model.to(device=device)

        optimizer = AdamW(model.parameters(), lr=conf["training.lr"], correct_bias=False)

        records = {
            # need to pair model and optimizer as optimizer need it to be reloaded
            "model_optimizer": QueryCodeSiameseModelAndAdamW(model, optimizer),
            "tokenizer": tokenizer,
            "common_tokens": common_tokens,
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
        languages, similarity, query_tokens, query_tokens_mask, code_tokens, code_tokens_mask, code_lang_weights = [
            t.to(self.device) for t in batch
        ]
        (query_embedding, code_embedding) = self.model(
            languages=languages,
            query_tokens=query_tokens,
            query_tokens_mask=query_tokens_mask,
            code_tokens=code_tokens,
            code_tokens_mask=code_tokens_mask,
            lang_weights=code_lang_weights,
        )
        per_sample_losses, similarity_scores = self.losses_scores_fn(
            query_embedding, code_embedding, similarity, code_lang_weights
        )
        avg_loss = torch.sum(per_sample_losses) / torch.sum(code_lang_weights)

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
        return self.tokenizer.encode_sentences(sentences, max_length)

    def tokenize_code_sentences(
        self, sentences: List[str], max_length: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        return self.tokenizer.encode_sentences(sentences, max_length)

    def tokenize_code_tokens(
        self, tokens: Iterable[List[str]], max_length: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        return self.tokenizer.encode_tokens(tokens, max_length)

    def build_lang_dataset(self, dataset_type: DatasetType) -> LangDataset:
        """Build language dataset using custom training context tokenizers"""
        common_toks: Dict[int, List[int]]

        if dataset_type == DatasetType.TRAIN:
            dirs = self.train_dirs
            name = f"train_{self.training_tokenizer_type}"
            data_params = self.train_data_params
            common_toks = {}
            if self.common_tokens is not None:
                for lang, toks in self.common_tokens.items():
                    lang_idx = data_params.lang_ids[lang]
                    lang_toks = [tok for tok, _ in toks]
                    lang_toks_idss, _ = self.tokenize_query_sentences(lang_toks)
                    tok_ids: List[int] = []
                    for lang_toks_ids in lang_toks_idss:
                        tok_ids.extend(lang_toks_ids.tolist())
                    common_toks[lang_idx] = tok_ids
        elif dataset_type == DatasetType.VAL:
            dirs = self.val_dirs
            name = f"val_{self.training_tokenizer_type}"
            data_params = self.val_data_params
            common_toks = {}

        elif dataset_type == DatasetType.TEST:
            dirs = self.test_dirs
            name = f"test_{self.training_tokenizer_type}"
            data_params = self.test_data_params
            common_toks = {}

        return build_lang_dataset_siamese_tokenizer(
            dirs=dirs,
            name=name,
            data_params=data_params,
            tokenizer=self.tokenizer,
            lang_token="<lg>",
            query_token="<qy>",
            fraction_using_func_name=data_params.fraction_using_func_name,
            query_random_token_frequency=data_params.query_random_token_frequency,
            common_tokens=common_toks,
            use_lang_weights=data_params.use_lang_weights,
            lang_ids=data_params.lang_ids,
            pickle_path=self.pickle_path,
            parallelize=self.train_data_params.parallelize,
        )

    def build_tokenizers(self, from_dataset_type: DatasetType) -> bool:
        if from_dataset_type == DatasetType.TRAIN:
            data_params = self.train_data_params
            dirs = self.train_dirs
        elif from_dataset_type == DatasetType.VAL:
            data_params = self.val_data_params
            dirs = self.val_dirs
        elif from_dataset_type == DatasetType.TEST:
            data_params = self.test_data_params
            dirs = self.test_dirs

        if data_params.use_subtokens:
            logger.info("Using SubTokenization")

        def sample_update(tpe: str, lang: str, tokens: List[str]) -> str:
            if data_params.use_subtokens:
                tokens = list(_to_subtoken_stream(tokens, mark_subtoken_end=False))
            if tpe == "code":
                return f"{lang} <lg> {' '.join(tokens)}\r\n"
            elif tpe == "query":
                return f"<qy> {' '.join(tokens)}\r\n"
            else:
                raise ValueError("tpe can be 'code' or 'query'")

        build_huggingface_bpetokenizers(
            dirs=dirs,
            data_params=data_params,
            build_path=self.tokenizers_build_path,
            token_path=self.tokenizers_token_files,
            sample_update=sample_update,
        )
        return True

    @classmethod
    def merge_contexts(
        cls, fresh_ctx: "QueryCodeSiameseCtx", restored_ctx: "QueryCodeSiameseCtx"
    ) -> "QueryCodeSiameseCtx":
        logger.debug("MERGE_CONTEXTS")
        fresh_ctx.model = restored_ctx.model
        fresh_ctx.tokenizer = restored_ctx.tokenizer
        return fresh_ctx


def load_tokenizers_from_hocon(conf: ConfigTree) -> Recordable:
    build_path = Path(conf["tokenizers.build_path"])

    if not os.path.exists(build_path):
        logger.error(
            f"Couldn't find {build_path} where tokenizers should have been built and stored... returning None tokenizer"
        )
        return NoneRecordable()

    records = RecordableMapping.load(build_path)
    if "tokenizer" in records:
        tokenizer = cast(TokenizerRecordable, records["tokenizer"])

        return tokenizer
    else:
        logger.error(f"Couldn't find 'tokenizer' recordables in path {build_path}")
        return NoneRecordable()


def load_common_tokens_from_hocon(conf: ConfigTree) -> Recordable:
    build_path = Path(conf["tokenizers.build_path"])
    if not os.path.exists(build_path):
        logger.error(
            f"Couldn't find {build_path} where tokenizers should have been built and stored... returning None tokenizer"
        )
        return NoneRecordable()

    records = RecordableMapping.load(build_path)

    if "common_tokens" in records:
        common_tokens = cast(DictRecordable, records["common_tokens"])

        return common_tokens
    else:
        logger.error(f"Couldn't find 'common_tokens' recordables in path {build_path}")
        return NoneRecordable()


def train_huggingface_bpetokenizers(
    data_params: DatasetParams, query_files: List[Path], lang_files: Dict[str, Path]
) -> TokenizerRecordable:
    logger.info(
        f"Building Siamese BPETokenizer from query_files {query_files} and lang_files {lang_files} with do_lowercase:{data_params.do_lowercase} special_tokens:{data_params.special_tokens}"
    )
    tokenizer = BPETokenizer()
    tokenizer.normalizer = BertNormalizer.new(
        clean_text=True, handle_chinese_chars=True, strip_accents=True, lowercase=data_params.do_lowercase
    )
    tokenizer.train(
        files=list(map(str, query_files + list(lang_files.values()))),
        vocab_size=data_params.vocab_size,
        special_tokens=data_params.special_tokens,
    )
    return HuggingfaceBPETokenizerRecordable(tokenizer)


def build_huggingface_bpetokenizers(
    dirs: List[Path],
    data_params: DatasetParams,
    sample_update: Callable[[str, str, List[str]], str],
    build_path: Union[str, Path],
    token_path: Union[str, Path],
) -> TokenizerRecordable:
    start = time.time()

    query_files, lang_files = build_huggingface_token_files(dirs, data_params, token_path, sample_update)
    tokenizer = train_huggingface_bpetokenizers(data_params, query_files, lang_files)
    end = time.time()

    time_p = end - start
    logger.info(f"tokenizer trainings took: {time_p} sec")

    os.makedirs(build_path, exist_ok=True)
    records = RecordableMapping({"tokenizer": tokenizer})
    records.save(build_path)

    # testing query_tokenizer
    txt = "This is a docstring".lower()
    encoded_ids, mask = tokenizer.encode_sentence(txt)
    logger.debug(f"encoded_ids {encoded_ids}")
    decoded = tokenizer.decode_sequence(encoded_ids)
    logger.debug(f"decoded {decoded}")
    logger.debug(f"txt {txt}")
    # assert decoded == txt

    return tokenizer
