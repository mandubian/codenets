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
from codenets.recordable import Recordable, RecordableMapping
from codenets.codesearchnet.dataset_utils import LangDataset
from codenets.codesearchnet.data import DatasetParams
from codenets.codesearchnet.training_ctx import CodeSearchTrainingContext, DatasetType
from codenets.codesearchnet.tokenizer_recs import TokenizerRecordable
from codenets.codesearchnet.training_ctx import ModelAndAdamWRecordable, default_sample_update

# from codenets.codesearchnet.tokenizer_recs import load_query_code_tokenizers_from_hocon_single_code_tokenizer
from codenets.codesearchnet.query_1_code_1.model import Query1Code1
from codenets.codesearchnet.query_1_code_1.dataset import build_lang_dataset_single_code_tokenizer
from codenets.codesearchnet.huggingface.tokenizer_recs import (
    HuggingfaceBPETokenizerRecordable,
    build_huggingface_token_files,
)


class Query1Code1ModelAndAdamW(ModelAndAdamWRecordable):
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

    model_type = Query1Code1

    def __init__(self, model: Query1Code1, optimizer: AdamW):
        super(Query1Code1ModelAndAdamW, self).__init__(model, optimizer)


class Query1Code1Ctx(CodeSearchTrainingContext):
    def __init__(self, records: Mapping[str, Recordable]):
        super(Query1Code1Ctx, self).__init__(records)
        self.tokenizers_build_path = Path(self.conf["tokenizers.build_path"])
        self.query_tokenizer = cast(TokenizerRecordable, records["query_tokenizer"])
        self.code_tokenizer = cast(TokenizerRecordable, records["code_tokenizer"])

        model_optimizer_rec = cast(Query1Code1ModelAndAdamW, records["model_optimizer"])
        self.model = model_optimizer_rec.model
        self.model = self.model.to(device=self.device)
        self.optimizer = model_optimizer_rec.optimizer
        # trick to pass params to refresh params on the right device
        # as there is no function for that in pytorch
        # need to find a nicer solution...
        self.optimizer.load_state_dict(self.optimizer.state_dict())

    @classmethod
    def from_hocon_custom(cls: Type["Query1Code1Ctx"], conf: ConfigTree) -> Mapping[str, Recordable]:
        res = load_tokenizers_from_hocon(conf)
        if res is not None:
            query_tokenizer, code_tokenizer = res
        else:
            #raise ValueError("Couldn't load Tokenizers from conf")
            query_tokenizer, code_tokenizer = None, None

        device = torch.device(conf["training.device"])
        model = Query1Code1.from_hocon(conf)
        model = model.to(device=device)

        optimizer = AdamW(model.parameters(), lr=conf["training.lr"], correct_bias=False)

        records = {
            # need to pair model and optimizer as optimizer need it to be reloaded
            "model_optimizer": Query1Code1ModelAndAdamW(model, optimizer),
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

    def forward(self, batch: List[Tensor], batch_idx: int) -> Tuple[Tensor, Tensor]:
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
        )
        batch_total_loss, similarity_scores = self.losses_scores_fn(
            query_embedding, code_embedding, similarity, code_lang_weights
        )

        return (batch_total_loss, similarity_scores)

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
            dirs = self.train_dirs
            name = f"train_{self.training_tokenizer_type}"
            data_params = self.train_data_params

        elif dataset_type == DatasetType.VAL:
            dirs = self.val_dirs
            name = f"val_{self.training_tokenizer_type}"
            data_params = self.val_data_params

        elif dataset_type == DatasetType.TEST:
            dirs = self.test_dirs
            name = f"test_{self.training_tokenizer_type}"
            data_params = self.test_data_params

        return build_lang_dataset_single_code_tokenizer(
            dirs=dirs,
            name=name,
            data_params=data_params,
            query_tokenizer=self.query_tokenizer,
            code_tokenizer=self.code_tokenizer,
            lang_token="<lg>",
            use_lang_weights=self.train_data_params.use_lang_weights,
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

        def sample_update(tpe: str, lang: str, tokens: List[str]) -> str:
            if tpe == "code":
                return f"{lang} <lg> {' '.join(tokens)}\r\n"
            else:
                return default_sample_update(tpe, lang, tokens)

        build_huggingface_bpetokenizers(
            dirs=dirs, data_params=data_params, output_path=self.tokenizers_build_path, sample_update=sample_update
        )
        return True


def load_tokenizers_from_hocon(conf: ConfigTree) -> Optional[Tuple[TokenizerRecordable, TokenizerRecordable]]:
    build_path = Path(conf["tokenizers.build_path"])

    if not os.path.exists(build_path):
        logger.error(f"Could find {build_path} where tokenizers should have been built and stored")
        return None

    records = RecordableMapping.load(build_path)
    if "query_tokenizer" in records and "code_tokenizer" in records:
        query_tokenizer = cast(TokenizerRecordable, records["query_tokenizer"])
        code_tokenizer = cast(TokenizerRecordable, records["code_tokenizer"])

        return query_tokenizer, code_tokenizer
    else:
        logger.error(f"Couldn't query_tokenizer/code_tokenizer recordables in path {build_path}")
        return None


def train_huggingface_bpetokenizers(
    data_params: DatasetParams, query_files: List[Path], lang_files: Dict[str, Path]
) -> Tuple[TokenizerRecordable, TokenizerRecordable]:
    logger.info(
        f"Building Query BPETokenizer from query_files {query_files} with do_lowercase:{data_params.do_lowercase} special_tokens:{data_params.special_tokens}"
    )
    query_tokenizer = BPETokenizer()
    query_tokenizer.normalizer = BertNormalizer.new(
        clean_text=True, handle_chinese_chars=True, strip_accents=True, lowercase=data_params.do_lowercase
    )
    query_tokenizer.train(
        files=list(map(str, query_files)), vocab_size=data_params.vocab_size, special_tokens=data_params.special_tokens
    )

    code_tokenizer = BPETokenizer()
    code_tokenizer.normalizer = BertNormalizer.new(
        clean_text=True, handle_chinese_chars=True, strip_accents=True, lowercase=data_params.do_lowercase
    )
    code_tokenizer.train(
        files=list(map(str, lang_files.values())),
        vocab_size=data_params.vocab_size,
        special_tokens=data_params.special_tokens,
    )

    return HuggingfaceBPETokenizerRecordable(query_tokenizer), HuggingfaceBPETokenizerRecordable(code_tokenizer)


def build_huggingface_bpetokenizers(
    dirs: List[Path],
    data_params: DatasetParams,
    output_path: Union[str, Path] = ".",
    sample_update: Callable[[str, str, List[str]], str] = default_sample_update,
) -> Tuple[TokenizerRecordable, TokenizerRecordable]:
    output_path = Path(output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    start = time.time()

    query_files, lang_files = build_huggingface_token_files(dirs, data_params, output_path, sample_update)
    query_tokenizer, code_tokenizer = train_huggingface_bpetokenizers(data_params, query_files, lang_files)
    #query_tokenizer_rec = HuggingfaceBPETokenizerRecordable(query_tokenizer)
    #code_tokenizer_rec = HuggingfaceBPETokenizerRecordable(code_tokenizer)
    end = time.time()

    time_p = end - start
    logger.info(f"query_tokenizer/code_tokenizer trainings took: {time_p} sec")

    records = RecordableMapping({"query_tokenizer": query_tokenizer, "code_tokenizer": code_tokenizer})
    records.save(output_path)

    # testing query_tokenizer
    txt = "This is a docstring".lower()
    encoded_ids, mask = query_tokenizer.encode_sentence(txt)
    logger.debug(f"encoded_ids {encoded_ids}")
    decoded = query_tokenizer.decode_sequence(encoded_ids)
    logger.debug(f"decoded {decoded}")
    logger.debug(f"txt {txt}")
    # assert decoded == txt

    return query_tokenizer, code_tokenizer
