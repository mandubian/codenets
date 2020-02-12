from typing import Iterable, List, Optional, Tuple, Union, Dict, cast, Callable, IO
import numpy as np
import os
from loguru import logger
from pathlib import Path
from transformers import PreTrainedTokenizer, BertTokenizer

from tokenizers import BPETokenizer
from codenets.recordable import Recordable, instance_full_classname, full_classname, RecordableMapping
from codenets.codesearchnet.data import DatasetParams
from codenets.codesearchnet.copied_code.metadata import Metadata, append_metadata, build_tokenizer_metadata
from codenets.codesearchnet.tokenizer_recs import TokenizerRecordable
from codenets.codesearchnet.copied_code.utils import read_file_samples
from codenets.utils import get_data_files_from_directory
from codenets.codesearchnet.training_ctx import default_sample_update


class PreTrainedTokenizerRecordable(TokenizerRecordable):
    def __init__(self, vocab: PreTrainedTokenizer):
        self.vocab = vocab

    def tokenize(self, text: str, **kwargs) -> List[str]:
        return self.vocab.tokenize(text)

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return self.vocab.convert_tokens_to_ids(tokens)

    def unk_token(self) -> str:
        return self.vocab.unk_token()

    # def pad_token(self) -> str:
    #     return self.vocab.pad_token()

    def encode_sentence(self, sentence: str, max_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        encoded = self.vocab.encode_plus(
            sentence,
            max_length=max_length,
            pad_to_max_length=max_length is not None,
            return_token_type_ids=False,
            return_attention_mask=True,
        )
        token_ids = np.array(encoded["input_ids"])
        token_mask = np.array(encoded["attention_mask"])
        return token_ids, token_mask

    def encode_sentences(
        self, sentences: List[str], max_length: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        encoded = self.vocab.batch_encode_plus(
            sentences,
            max_length=max_length,
            pad_to_max_length=max_length is not None,
            return_token_type_ids=False,
            return_attention_mask=True,
        )
        token_ids = np.array(encoded["input_ids"])
        token_mask = np.array(encoded["attention_mask"])
        return (token_ids, token_mask)

    def encode_tokens(
        self, tokens: Iterable[List[str]], max_length: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        encoded = self.vocab.batch_encode_plus(
            tokens,
            max_length=max_length,
            pad_to_max_length=max_length is not None,
            return_token_type_ids=False,
            return_attention_mask=True,
        )
        token_ids = np.array(encoded["input_ids"])
        token_mask = np.array(encoded["attention_mask"])
        return (token_ids, token_mask)

    def decode_sequence(self, tokens_sequence: List[int]) -> str:
        return self.vocab.decode(tokens_sequence)

    def decode_sequences(self, tokens_sequences: Iterable[List[int]]) -> List[str]:
        return self.vocab.decode_batch(tokens_sequences)


class BertTokenizerRecordable(PreTrainedTokenizerRecordable):
    def __init__(self, vocab: BertTokenizer):
        super(BertTokenizerRecordable, self).__init__(vocab)

    def save(self, output_dir: Union[Path, str]) -> bool:
        full_dir = Path(output_dir) / instance_full_classname(self)
        logger.debug(f"Saving BertTokenizerRecordable to {full_dir}")
        os.makedirs(full_dir, exist_ok=True)
        self.vocab.save_pretrained(full_dir)
        return True

    @classmethod
    def load(cls, restore_dir: Union[Path, str]) -> "BertTokenizerRecordable":
        full_dir = Path(restore_dir) / full_classname(cls)
        logger.debug(f"Loading BertTokenizerRecordable from {full_dir}")
        vocab = BertTokenizer.from_pretrained(str(full_dir))
        return BertTokenizerRecordable(vocab)


class HuggingfaceBPETokenizerRecordable(TokenizerRecordable):
    def __init__(self, vocab: BPETokenizer):
        self.vocab = vocab

    def tokenize(self, text: str, **kwargs) -> List[str]:
        return self.vocab.encode(text).tokens

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.vocab.token_to_id(tok) for tok in tokens]
        # return self.vocab.convert_tokens_to_ids(tokens)

    def unk_token(self) -> str:
        # no access to that in
        return "<unk>"

    # def pad_token(self) -> str:
    #     return self.vocab.pad_token()

    def encode_sentence(self, sentence: str, max_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        enc = self.vocab.encode(sentence)
        if max_length is not None:
            enc.truncate(max_length)
            enc.pad(max_length)
        return np.array(enc.ids), np.array(enc.attention_mask)

    def encode_sentences(
        self, sentences: List[str], max_length: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        encs = self.vocab.encode_batch(sentences)
        if max_length is not None:
            for enc in encs:
                enc.truncate(max_length)
                enc.pad(max_length)
        tokens_ids = np.array([enc.ids for enc in encs])
        attention_mask = np.array([enc.attention_mask for enc in encs])
        return (tokens_ids, attention_mask)

    def encode_tokens(
        self, tokens: Iterable[List[str]], max_length: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        # hack...
        sentences = [" ".join(toks) for toks in tokens]
        return self.encode_sentences(sentences, max_length)

    def decode_sequence(self, tokens_sequence: List[int]) -> str:
        return self.vocab.decode(tokens_sequence)

    def decode_sequences(self, tokens_sequences: Iterable[List[int]]) -> List[str]:
        return self.vocab.decode_batch(tokens_sequences)

    def save(self, output_dir: Union[Path, str]) -> bool:
        full_dir = Path(output_dir) / instance_full_classname(self)
        logger.debug(f"HuggingfaceBPETokenizerRecordable - Saving to {full_dir}")
        os.makedirs(full_dir, exist_ok=True)

        self.vocab._tokenizer.model.save(str(full_dir), name=str(instance_full_classname(self)))
        return True

    @classmethod
    def load(cls, restore_dir: Union[Path, str]) -> "HuggingfaceBPETokenizerRecordable":
        full_dir = Path(restore_dir) / full_classname(cls)
        logger.debug(f"HuggingfaceBPETokenizerRecordable - Loading from {full_dir}")
        # vocab = pickle.load(open(full_dir / "vocab.pth", "rb"))
        # model = models.BPE.from_files(
        #     full_dir / f"{full_classname(cls)}_vocab.json", full_dir / f"{full_classname(cls)}_merges.json"
        # )
        tokenizer = BPETokenizer(
            vocab_file=str(full_dir / f"{full_classname(cls)}-vocab.json"),
            merges_file=str(full_dir / f"{full_classname(cls)}-merges.txt"),
        )
        return HuggingfaceBPETokenizerRecordable(tokenizer)


def build_huggingface_token_files(
    data_dirs: List[Path],
    data_params: DatasetParams,
    output_path: Union[Path, str],
    sample_update: Callable[[str, str, List[str]], str] = default_sample_update,
) -> Tuple[List[Path], Dict[str, Path]]:
    tokenizers_path = Path(output_path)
    os.makedirs(tokenizers_path, exist_ok=True)
    # build files of strings
    lang_ios: Dict[str, Tuple[IO[str], IO[str]]] = {}

    query_files: List[Path] = []
    lang_files: Dict[str, Path] = {}
    for (idx, file_path) in enumerate(get_data_files_from_directory(data_dirs)):
        logger.info(f"Reading {file_path}")
        for raw_sample in read_file_samples(file_path):
            lang = raw_sample["language"]
            if lang not in lang_ios:
                query_file = tokenizers_path / f"{lang}_query.txt"
                code_file = tokenizers_path / f"{lang}_code.txt"
                lang_ios[lang] = (open(query_file, "w"), open(code_file, "w"))
                query_files.append(query_file)
                lang_files[lang] = code_file
            lang_ios[lang][0].write(sample_update("query", lang, raw_sample["docstring_tokens"]))
            lang_ios[lang][1].write(sample_update("code", lang, raw_sample["code_tokens"]))

    return query_files, lang_files
