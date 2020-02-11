# This code is nearly 100% copied from original repo

from dataclasses import dataclass, fields as datafields
import numpy as np

from typing import Dict, TypeVar, Type, List


@dataclass
class DatasetParams:
    """Description of parameters of a CodeSearchnet dataset"""

    fraction_using_func_name: float
    min_len_func_name_for_query: int
    use_subtokens: bool
    mark_subtoken_end: bool
    code_max_num_tokens: int
    query_max_num_tokens: int
    use_bpe: bool
    vocab_size: int
    pct_bpe: float
    vocab_count_threshold: int
    lang_ids: Dict[str, int]
    do_lowercase: bool
    special_tokens: List[str]
    parallelize: bool


T_InputFeatures = TypeVar("T_InputFeatures", bound="InputFeatures")


@dataclass
class InputFeatures:
    """Structure gathering query and code tokens/mask after passing through tokenizer"""

    language: int
    similarity: float
    query_tokens: np.ndarray
    query_tokens_mask: np.ndarray
    # query_tokens_length: int

    code_tokens: np.ndarray
    code_tokens_mask: np.ndarray
    # code_tokens_length: int

    # @classmethod
    # def from_dict(cls: Type[T_InputFeatures], dikt) -> T_InputFeatures:
    #     """Create an instance of dataclass from a dict
    #     return cls(**dikt)


def dataclass_from_dict(klass, dikt):
    """Load any dataclass from a dict"""
    fieldtypes = {f.name: f.type for f in datafields(klass)}
    return klass(**{f: dataclass_from_dict(fieldtypes[f], dikt[f]) for f in dikt})
