from abc import abstractmethod

from typing import Iterable, List, Optional, Tuple, Dict, cast
import numpy as np
import os
from loguru import logger
from pathlib import Path
import pickle

import time

from pyhocon import ConfigTree
from codenets.recordable import Recordable, RecordableMapping, DictRecordable
from codenets.codesearchnet.data import DatasetParams
from codenets.codesearchnet.copied_code.metadata import Metadata, append_metadata, build_tokenizer_metadata


class TokenizerRecordable(Recordable):
    @abstractmethod
    def tokenize(self, text: str, **kwargs) -> List[str]:
        pass

    @abstractmethod
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        pass

    @abstractmethod
    def unk_token(self) -> str:
        pass

    # @abstractmethod
    # def pad_token(self) -> str:
    #     pass

    @abstractmethod
    def encode_sentence(self, sentence: str, max_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def encode_sentences(
        self, sentences: List[str], max_length: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def encode_tokens(
        self, tokens: Iterable[List[str]], max_length: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def decode_sequence(self, tokens_sequence: List[int]) -> str:
        pass

    @abstractmethod
    def decode_sequences(self, tokens_sequences: Iterable[List[int]]) -> List[str]:
        pass

    @abstractmethod
    def add_special_tokens(self, special_tokens: List[str]) -> bool:
        pass


def build_most_common_tokens(
    data_dirs: List[Path],
    data_params: DatasetParams,
    build_path: Path,
    max_files_per_dir: Optional[int] = None,
    parallelize: bool = True,
) -> Dict[str, List[Tuple[str, int]]]:

    start = time.time()

    logger.info(f"Build metadata for {data_dirs}")

    _, code_language_metadata_lists = build_tokenizer_metadata(
        data_dirs=data_dirs,
        max_files_per_dir=max_files_per_dir,
        parallelize=parallelize,
        use_subtokens=data_params.use_subtokens,
        mark_subtoken_end=data_params.mark_subtoken_end,
    )

    logger.info("Merging metadata")

    # merge metadata if necessary
    per_code_language_metadata: Dict[str, Metadata] = {}
    for (language, raw_per_language_metadata) in code_language_metadata_lists.items():
        logger.info(f"Build vocabulary for {language}")
        per_code_language_metadata[language] = append_metadata(
            "code",
            vocab_size=data_params.vocab_size,
            vocab_count_threshold=data_params.vocab_count_threshold,
            pct_bpe=data_params.pct_bpe,
            raw_metadata_list=raw_per_language_metadata,
        )
    common_tokens: Dict[str, List[Tuple[str, int]]] = {}
    for (language, md) in per_code_language_metadata.items():
        common_tokens[language] = md.common_tokens

    end = time.time()

    time_p = end - start
    logger.info(f"Most Common Tokens: {time_p} sec")

    pickle.dump(common_tokens, open("./checkpoints/tmp_common_tokens.p", "wb"))

    common_tokens_dict = DictRecordable(common_tokens)
    os.makedirs(build_path, exist_ok=True)
    records = RecordableMapping({"common_tokens": common_tokens_dict})
    records.save(build_path)

    return common_tokens_dict


def load_query_code_tokenizers_from_hocon(conf: ConfigTree) -> Optional[Tuple[TokenizerRecordable, RecordableMapping]]:
    build_path = Path(conf["tokenizers.build_path"])

    if not os.path.exists(build_path):
        logger.error(f"Could find {build_path} where tokenizers should have been built and stored")
        return None

    records = RecordableMapping.load(build_path)
    if "query_tokenizer" in records and "code_tokenizers" in records:
        query_tokenizer = cast(TokenizerRecordable, records["query_tokenizer"])
        code_tokenizers = cast(RecordableMapping, records["code_tokenizers"])

        return query_tokenizer, code_tokenizers
    else:
        logger.error(f"Couldn't query_tokenizer/code_tokenizers recordables in path {build_path}")
        return None
