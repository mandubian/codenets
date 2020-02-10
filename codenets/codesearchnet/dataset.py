# This file mixes code from oroginal CodeSearchNet project ported to my own
# vision of programming with types and python and pytorch...

import os
import sys

from pathos.pools import ProcessPool
from abc import abstractmethod
import random

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, TypeVar, Callable, cast, Iterator

import numpy as np
import itertools
import torch
from torch.utils.data import Dataset, TensorDataset, ConcatDataset, RandomSampler, Sampler

from dpu_utils.codeutils import split_identifier_into_parts

from loguru import logger
from pathlib import Path
import pickle
import functools

from enum import Enum

from codenets.codesearchnet.data import DatasetParams, InputFeatures
from codenets.utils import _to_subtoken_stream, get_data_files_from_directory
from codenets.codesearchnet.original.utils import read_file_samples
from codenets.codesearchnet.tokenizer_recs import TokenizerRecordable
from codenets.codesearchnet.original.metadata import QueryType


class DatasetType(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def convert_and_pad_token_sequence(
    tokenizer: TokenizerRecordable, token_sequence: List[str], output_tensor_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tensorise token sequence with padding; returning a mask for used elements as well.

    Args:
        tokenizer: Tokenizer.
        token_sequence: List of tokens in string form
        output_tensor_size: Size of the resulting tensor (i.e., length up which we pad / down to which we truncate.
        pad_from_left: Indicate if we are padding/truncating on the left side of string. [Default: False]

    Returns:
        Pair of numpy arrays. First is the actual tensorised token sequence, the second is a masking tensor
        that is 1.0 for those token indices that are actually used.
    """
    token_ids, token_mask = tokenizer.encode_tokens([token_sequence], max_length=output_tensor_size)
    return token_ids[0], token_mask[0]


def load_data_from_sample(
    language: str,
    encoder_label: str,
    data_to_load: Any,
    function_name: Optional[str],
    tokenizer: TokenizerRecordable,
    fraction_using_func_name: float,
    min_len_func_name_for_query: int,
    use_subtokens: bool,
    mark_subtoken_end: bool,
    max_num_tokens: int,
    lang_token: str = "<lg>",
) -> Optional[Dict[str, np.ndarray]]:
    """
    Save two versions of both the code and the query: one using the docstring as the query and the other using the
    function-name as the query, and replacing the function name in the code with an out-of-vocab token.
    Sub-tokenizes, converts, and pads both versions, and rejects empty samples.
    """
    result_holder: Dict[str, Any] = {}
    # Save the two versions of the code and query:
    data_holder = {QueryType.DOCSTRING.value: data_to_load, QueryType.FUNCTION_NAME.value: None}
    # Skip samples where the function name is very short, because it probably has too little information
    # to be a good search query.
    if fraction_using_func_name > 0.0 and function_name and len(function_name) >= min_len_func_name_for_query:
        if encoder_label == "query":
            # Set the query tokens to the function name, broken up into its sub-tokens:
            data_holder[QueryType.FUNCTION_NAME.value] = split_identifier_into_parts(function_name)
        elif encoder_label == "code":
            # In the code, replace the function name with the out-of-vocab token everywhere it appears:
            data_holder[QueryType.FUNCTION_NAME.value] = [
                tokenizer.unk_token() if token == function_name else token for token in data_to_load
            ]
    else:
        return None

    # Sub-tokenize, convert, and pad both versions:
    for key, data in data_holder.items():
        # if hyperparameters[f"{encoder_label}_use_subtokens"]:
        if use_subtokens:
            data = _to_subtoken_stream(data, mark_subtoken_end=mark_subtoken_end)
        tokens, tokens_mask = convert_and_pad_token_sequence(
            tokenizer=tokenizer, token_sequence=list(data), output_tensor_size=max_num_tokens
        )
        # Note that we share the result_holder with different encoders, and so we need to make our identifiers
        # unique-ish
        result_holder[f"{encoder_label}_tokens_{key}"] = tokens
        result_holder[f"{encoder_label}_tokens_mask_{key}"] = tokens_mask

    if (
        result_holder[f"{encoder_label}_tokens_mask_{QueryType.DOCSTRING.value}"] is None
        or int(np.sum(result_holder[f"{encoder_label}_tokens_mask_{QueryType.DOCSTRING.value}"])) == 0
    ):
        return result_holder

    return result_holder


# ugly type inheritede from original CodeSearchNet code...
# will replace that later
Samples = Iterable[Dict[str, Union[str, int, np.ndarray]]]


def parse_data_file(
    data_file: Path,
    data_params: DatasetParams,
    query_tokenizer: TokenizerRecordable,
    per_code_language_tokenizers: Dict[str, TokenizerRecordable],
) -> Tuple[str, int, Samples]:
    logger.info(f"Reading samples from {data_file}")
    filename = os.path.basename(data_file)
    file_language = filename.split("_")[0]

    samples = list(read_file_samples(data_file))

    ds: List[Dict[str, Union[str, int]]] = []
    for raw_sample in samples:
        language = raw_sample["language"]
        if language.startswith("python"):  # In some datasets, we use 'python-2.7' and 'python-3'
            language = "python"

        if language != file_language:
            logger.error(f"file with different language {language} from filename {file_language}")
            sys.exit(f"file with multiple language {language} from filename {file_language}")

        # the load_data_from_sample method call places processed data into sample, and
        # returns a boolean flag indicating if sample should be used
        function_name = raw_sample.get("func_name")
        data_code = load_data_from_sample(
            language=language,
            encoder_label="code",
            data_to_load=raw_sample["code_tokens"],
            function_name=function_name,
            tokenizer=per_code_language_tokenizers[language],
            fraction_using_func_name=data_params.fraction_using_func_name,
            min_len_func_name_for_query=data_params.min_len_func_name_for_query,
            use_subtokens=data_params.use_subtokens,
            mark_subtoken_end=data_params.mark_subtoken_end,
            max_num_tokens=data_params.code_max_num_tokens,
        )

        data_query = load_data_from_sample(
            language=language,
            encoder_label="query",
            data_to_load=[d.lower() for d in raw_sample["docstring_tokens"]],
            function_name=function_name,
            tokenizer=query_tokenizer,
            fraction_using_func_name=data_params.fraction_using_func_name,
            min_len_func_name_for_query=data_params.min_len_func_name_for_query,
            use_subtokens=data_params.use_subtokens,
            mark_subtoken_end=data_params.mark_subtoken_end,
            max_num_tokens=data_params.query_max_num_tokens,
        )

        if data_code is not None and data_query is not None:
            d = {"language": language, "similarity": 1, **data_code, **data_query}
            ds.append(d)

    logger.debug(f"Parsed file {data_file}: language {file_language} [{len(ds)} samples]")

    return (file_language, len(ds), ds)


T = TypeVar("T")


def load_data_from_files(
    data_files: Iterable[Path],
    data_params: DatasetParams,
    query_tokenizer: TokenizerRecordable,
    per_code_language_tokenizers: Dict[str, TokenizerRecordable],
    # humm that is not very nice type signature... need to create interface for that
    parse_callback: Callable[
        [Path, DatasetParams, TokenizerRecordable, Dict[str, TokenizerRecordable]], Tuple[str, int, Iterable[T]]
    ],
    parallelize: bool = True,
) -> Dict[str, Tuple[int, Iterable[T]]]:
    tasks_as_args = [
        [data_file, data_params, query_tokenizer, per_code_language_tokenizers] for data_file in data_files
    ]

    if parallelize:
        pool = ProcessPool()

        # needed that hack to work... issues with serialization of classes
        # doesn't work with basic multiprocessing so needed pathos
        def cb(x):
            return parse_callback(*x)

        per_file_results = list(pool.map(cb, tasks_as_args))
    else:
        per_file_results = [parse_callback(*task_args) for task_args in tasks_as_args]  # type: ignore

    lang_samples_iter: Dict[str, Tuple[int, List[Iterable[T]]]] = {}
    for (lang, lg, samples_iter) in per_file_results:
        if lang not in lang_samples_iter:
            lang_samples_iter[lang] = (0, [])
        (lg0, iters) = lang_samples_iter[lang]
        iters.append(samples_iter)
        lang_samples_iter[lang] = (lg0 + lg, iters)

    lang_samples: Dict[str, Tuple[int, Iterable[T]]] = {}
    for (lang, (lg, iters)) in lang_samples_iter.items():
        lang_samples[lang] = (lg, itertools.chain(*iters))

    return lang_samples


def load_data_from_dirs(
    data_dirs: List[Path],
    query_tokenizer: TokenizerRecordable,
    per_code_language_tokenizers: Dict[str, TokenizerRecordable],
    data_params: DatasetParams,
    parse_callback: Callable[
        [Path, DatasetParams, TokenizerRecordable, Dict[str, TokenizerRecordable]], Tuple[str, int, Iterable[T]]
    ],
    max_files_per_dir: Optional[int] = None,
    parallelize: bool = True,
) -> Dict[str, Tuple[int, Iterable[T]]]:
    return load_data_from_files(
        data_files=list(get_data_files_from_directory(data_dirs, max_files_per_dir)),
        data_params=data_params,
        query_tokenizer=query_tokenizer,
        per_code_language_tokenizers=per_code_language_tokenizers,
        parse_callback=parse_callback,
        parallelize=parallelize,
    )


def convert_and_pad_token_sequence_single_tokenizer(
    tokenizer: TokenizerRecordable, token_sequence: List[str], output_tensor_size: int, language: str, lang_token: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tensorise token sequence with padding; returning a mask for used elements as well.

    Args:
        tokenizer: Tokenizer.
        token_sequence: List of tokens in string form
        output_tensor_size: Size of the resulting tensor (i.e., length up which we pad / down to which we truncate.
        pad_from_left: Indicate if we are padding/truncating on the left side of string. [Default: False]

    Returns:
        Pair of numpy arrays. First is the actual tensorised token sequence, the second is a masking tensor
        that is 1.0 for those token indices that are actually used.
    """
    token_sequence = [language, lang_token] + token_sequence
    token_ids, token_mask = tokenizer.encode_tokens([token_sequence], max_length=output_tensor_size)
    return token_ids[0], token_mask[0]


def load_data_from_sample_with_lang(
    language: str,
    encoder_label: str,
    data_to_load: Any,
    function_name: Optional[str],
    tokenizer: TokenizerRecordable,
    fraction_using_func_name: float,
    min_len_func_name_for_query: int,
    use_subtokens: bool,
    mark_subtoken_end: bool,
    max_num_tokens: int,
    lang_token: str,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Save two versions of both the code and the query: one using the docstring as the query and the other using the
    function-name as the query, and replacing the function name in the code with an out-of-vocab token.
    Sub-tokenizes, converts, and pads both versions, and rejects empty samples.
    """
    result_holder: Dict[str, Any] = {}
    # Save the two versions of the code and query:
    data_holder = {QueryType.DOCSTRING.value: data_to_load, QueryType.FUNCTION_NAME.value: None}
    # Skip samples where the function name is very short, because it probably has too little information
    # to be a good search query.
    if fraction_using_func_name > 0.0 and function_name and len(function_name) >= min_len_func_name_for_query:
        if encoder_label == "query":
            # Set the query tokens to the function name, broken up into its sub-tokens:
            data_holder[QueryType.FUNCTION_NAME.value] = split_identifier_into_parts(function_name)
        elif encoder_label == "code":
            # In the code, replace the function name with the out-of-vocab token everywhere it appears:
            data_holder[QueryType.FUNCTION_NAME.value] = [
                tokenizer.unk_token() if token == function_name else token for token in data_to_load
            ]
    else:
        return None

    # Sub-tokenize, convert, and pad both versions:
    for key, data in data_holder.items():
        # if hyperparameters[f"{encoder_label}_use_subtokens"]:
        if use_subtokens:
            data = _to_subtoken_stream(data, mark_subtoken_end=mark_subtoken_end)
        tokens, tokens_mask = convert_and_pad_token_sequence_single_tokenizer(
            tokenizer=tokenizer,
            token_sequence=list(data),
            output_tensor_size=max_num_tokens,
            language=language,
            lang_token=lang_token,
        )
        # Note that we share the result_holder with different encoders, and so we need to make our identifiers
        # unique-ish
        result_holder[f"{encoder_label}_tokens_{key}"] = tokens
        result_holder[f"{encoder_label}_tokens_mask_{key}"] = tokens_mask

    if (
        result_holder[f"{encoder_label}_tokens_mask_{QueryType.DOCSTRING.value}"] is None
        or int(np.sum(result_holder[f"{encoder_label}_tokens_mask_{QueryType.DOCSTRING.value}"])) == 0
    ):
        return result_holder

    return result_holder


def parse_data_file_single_code_tokenizer(
    data_file: Path,
    data_params: DatasetParams,
    query_tokenizer: TokenizerRecordable,
    code_tokenizer: TokenizerRecordable,
    lang_token: str,
) -> Tuple[str, int, Samples]:
    logger.info(f"Reading samples from {data_file}")
    filename = os.path.basename(data_file)
    file_language = filename.split("_")[0]

    samples = list(read_file_samples(data_file))

    ds: List[Dict[str, Union[str, int]]] = []
    for raw_sample in samples:
        language = raw_sample["language"]
        if language.startswith("python"):  # In some datasets, we use 'python-2.7' and 'python-3'
            language = "python"

        if language != file_language:
            logger.error(f"file with different language {language} from filename {file_language}")
            sys.exit(f"file with multiple language {language} from filename {file_language}")

        # the load_data_from_sample method call places processed data into sample, and
        # returns a boolean flag indicating if sample should be used
        function_name = raw_sample.get("func_name")
        data_code = load_data_from_sample_with_lang(
            language=language,
            encoder_label="code",
            data_to_load=raw_sample["code_tokens"],
            function_name=function_name,
            tokenizer=code_tokenizer,
            fraction_using_func_name=data_params.fraction_using_func_name,
            min_len_func_name_for_query=data_params.min_len_func_name_for_query,
            use_subtokens=data_params.use_subtokens,
            mark_subtoken_end=data_params.mark_subtoken_end,
            max_num_tokens=data_params.code_max_num_tokens,
            lang_token=lang_token,
        )

        # query doesn't use the language
        data_query = load_data_from_sample(
            language=language,
            encoder_label="query",
            data_to_load=[d.lower() for d in raw_sample["docstring_tokens"]],
            function_name=function_name,
            tokenizer=query_tokenizer,
            fraction_using_func_name=data_params.fraction_using_func_name,
            min_len_func_name_for_query=data_params.min_len_func_name_for_query,
            use_subtokens=data_params.use_subtokens,
            mark_subtoken_end=data_params.mark_subtoken_end,
            max_num_tokens=data_params.query_max_num_tokens,
        )

        if data_code is not None and data_query is not None:
            d = {"language": language, "similarity": 1, **data_code, **data_query}
            ds.append(d)

    logger.debug(f"Parsed file {data_file}: language {file_language} [{len(ds)} samples]")

    return (file_language, len(ds), ds)


T_Single = TypeVar("T_Single")


def load_data_from_files_single_tokenizer(
    data_files: Iterable[Path],
    data_params: DatasetParams,
    query_tokenizer: TokenizerRecordable,
    code_tokenizer: TokenizerRecordable,
    # humm that is not very nice type signature... need to create interface for that
    parse_callback: Callable[
        [Path, DatasetParams, TokenizerRecordable, TokenizerRecordable], Tuple[str, int, Iterable[T_Single]]
    ],
    parallelize: bool = True,
) -> Dict[str, Tuple[int, Iterable[T_Single]]]:
    tasks_as_args = [[data_file, data_params, query_tokenizer, code_tokenizer] for data_file in data_files]

    if parallelize:
        pool = ProcessPool()

        # needed that hack to work... issues with serialization of classes
        # doesn't work with basic multiprocessing so needed pathos
        def cb(x):
            return parse_callback(*x)

        per_file_results = list(pool.map(cb, tasks_as_args))
    else:
        per_file_results = [parse_callback(*task_args) for task_args in tasks_as_args]  # type: ignore

    lang_samples_iter: Dict[str, Tuple[int, List[Iterable[T_Single]]]] = {}
    for (lang, lg, samples_iter) in per_file_results:
        if lang not in lang_samples_iter:
            lang_samples_iter[lang] = (0, [])
        (lg0, iters) = lang_samples_iter[lang]
        iters.append(samples_iter)
        lang_samples_iter[lang] = (lg0 + lg, iters)

    lang_samples: Dict[str, Tuple[int, Iterable[T_Single]]] = {}
    for (lang, (lg, iters)) in lang_samples_iter.items():
        lang_samples[lang] = (lg, itertools.chain(*iters))

    return lang_samples


def load_data_from_dirs_single_code_tokenizer(
    data_dirs: List[Path],
    query_tokenizer: TokenizerRecordable,
    code_tokenizer: TokenizerRecordable,
    data_params: DatasetParams,
    parse_callback: Callable[
        [Path, DatasetParams, TokenizerRecordable, TokenizerRecordable], Tuple[str, int, Iterable[T_Single]]
    ],
    max_files_per_dir: Optional[int] = None,
    parallelize: bool = True,
) -> Dict[str, Tuple[int, Iterable[T_Single]]]:
    return load_data_from_files_single_tokenizer(
        data_files=list(get_data_files_from_directory(data_dirs, max_files_per_dir)),
        data_params=data_params,
        query_tokenizer=query_tokenizer,
        code_tokenizer=code_tokenizer,
        parse_callback=parse_callback,
        parallelize=parallelize,
    )


class ConcatNamedDataset(Dataset):
    @abstractmethod
    def get_datasets_nb(self) -> int:
        pass

    @abstractmethod
    def get_dataset_names(self) -> List[str]:
        pass

    @abstractmethod
    def get_datasets_info_by_desc_count(self) -> List[Tuple[int, str, int]]:
        pass

    @abstractmethod
    def get_lang_id(self, name: str) -> int:
        pass

    @abstractmethod
    def get_dataset_by_lang_id(self, lang_id: int) -> Dataset:
        pass

    @abstractmethod
    def get_lang_id_for_sample_index(self, idx: int) -> int:
        pass

    @abstractmethod
    def get_cumulative_sizes(self) -> List[int]:
        pass


class LangDataset(ConcatNamedDataset):
    """
    Dataset wrapping language features.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, lang_features: Dict[str, Tuple[int, Iterable[InputFeatures]]], lang_ids: Dict[str, int]):
        self.langs: List[str] = list(lang_features.keys())
        self.lang_ids = lang_ids
        self.datasets: List[TensorDataset] = []
        self.datasets_len: List[Tuple[int, str, int]] = []
        self.lang_indexes: Dict[int, int] = {}
        logger.info("Concatenating Datasets")
        lang_features_sorted = sorted(lang_features.items(), key=lambda lf: lf[1][0], reverse=True)
        for idx, (lang, (nb, features)) in enumerate(lang_features_sorted):
            lang_id = self.lang_ids[lang]
            self.lang_indexes[lang_id] = idx
            logger.info(f"Adding Language {lang} id:{idx} lang_id:{lang_id} [{nb} samples]")
            self.datasets_len.append((lang_id, lang, nb))
            ds = TensorDataset(*self.__tensorize_features(features))
            self.datasets.append(ds)

        # self.datasets = sorted(datasets, key=lambda d: len(d), reverse=True)
        self.concat_dataset: ConcatDataset = ConcatDataset(self.datasets)
        logger.info(f"Concat_dataset [{len(self.concat_dataset)} samples]")

    def __tensorize_features(self, features: Iterable[InputFeatures]) -> Tuple:
        # logger.debug(f"query_tokens {list(features)[0].query_tokens}")
        # logger.debug(f"query_tokens_mask {list(features)[0].query_tokens_mask}")
        # logger.debug(f"code_tokens {list(features)[0].code_tokens}")
        # logger.debug(f"code_tokens_mask {list(features)[0].code_tokens_mask}")
        all_query_tokens = torch.as_tensor([f.query_tokens for f in features], dtype=torch.long)
        all_query_tokens_mask = torch.as_tensor([f.query_tokens_mask for f in features], dtype=torch.long)
        all_code_tokens = torch.as_tensor([f.code_tokens for f in features], dtype=torch.long)
        all_code_tokens_mask = torch.as_tensor([f.code_tokens_mask for f in features], dtype=torch.long)
        all_languages = torch.as_tensor([f.language for f in features], dtype=torch.int8)
        all_similarity = torch.as_tensor([f.similarity for f in features], dtype=torch.float)
        return (
            all_languages,
            all_similarity,
            all_query_tokens,
            all_query_tokens_mask,
            all_code_tokens,
            all_code_tokens_mask,
        )

    def get_datasets_nb(self) -> int:
        return len(self.datasets)

    def get_lang_id(self, name: str) -> int:
        return self.lang_ids[name]

    def get_dataset_by_lang_id(self, lang_id: int) -> Dataset:
        return self.datasets[self.lang_indexes[lang_id]]

    def get_dataset_names(self) -> List[str]:
        return self.langs

    def get_datasets_info_by_desc_count(self) -> List[Tuple[int, str, int]]:
        return self.datasets_len

    def get_lang_id_for_sample_index(self, idx: int) -> int:
        return self.concat_dataset[idx][0].item()

    def get_cumulative_sizes(self) -> List[int]:
        return self.concat_dataset.cumulative_sizes

    def __getitem__(self, idx: int):
        """Get item"""
        return self.concat_dataset[idx]

    def __len__(self):
        """Get length"""
        return len(self.concat_dataset)


class BalancedBatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """Iterate over tasks and provide a balanced batch per task in each mini-batch"""

    def __init__(self, dataset: ConcatNamedDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = dataset.get_datasets_nb()

        self.samplers_list: List[Sampler] = []
        self.sampler_iterators: List[Iterator] = []
        self.datasets_length: List[int] = []  # as we get datasets in descending size, 0 is the largest dataset
        self.datasets_by_desc_size = self.dataset.get_datasets_info_by_desc_count()

        for idx, (lang_id, lang, dataset_count) in enumerate(self.datasets_by_desc_size):
            cur_dataset = self.dataset.get_dataset_by_lang_id(lang_id)
            logger.debug(
                f"Sampling batches from Dataset[lang_id:{lang_id}, count:{dataset_count}, len:{len(cur_dataset)} lang:{lang}]"
            )
            sampler = RandomSampler(cur_dataset)
            # if idx == 0:
            #     # the first dataset is kept at RandomSampler
            #     sampler = RandomSampler(cur_dataset)
            # else:
            #     # the second unbalanced dataset is changed
            #     def get_label(idx: int) -> str:
            #         return str(self.dataset.get_dataset_index_for_sample_index(idx))

            #     sampler = ImbalancedDatasetSampler(cur_dataset, callback_get_label=get_label)
            #     sampler = RandomSampler(cur_dataset)

            self.samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            self.sampler_iterators.append(cur_sampler_iterator)
            self.datasets_length.append(dataset_count)

        self.generate_sample_list()

    def generate_sample_list(self):
        for i in range(len(self.samplers_list)):
            self.sampler_iterators[i] = self.samplers_list[i].__iter__()
        push_index_val = [0] + self.dataset.get_cumulative_sizes()[:-1]
        logger.debug(f"push_index_val {push_index_val}")
        # largest_dataset_index = torch.argmax(torch.as_tensor(datasets_length)).item()
        # largest_dataset_index = datasets_by_desc_size[0][0]
        largest_dataset_index = 0
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        # epoch_samples = datasets_length[largest_dataset_index] * self.number_of_datasets
        epoch_samples = self.datasets_length[largest_dataset_index]
        # step = self.batch_size * self.number_of_datasets
        step = self.batch_size
        samples_to_grab = self.batch_size

        logger.debug(f"Generating batch of size {self.batch_size} from {epoch_samples} samples")
        final_samples_list = []  # this is a list of indexes from the combined dataset
        for epoch_idx in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                # print(f"changing dataset {i}")
                cur_batch_sampler = self.sampler_iterators[i]
                cur_samples = []
                for j in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        if i == largest_dataset_index:
                            # largest dataset iterator is done we can break
                            samples_to_grab = len(cur_samples)  # adjusting the samples_to_grab
                            break  # got to the end of iterator - extend final list and continue to next task
                        else:
                            # restart the iterator - we want more samples until finishing with the largest dataset
                            self.sampler_iterators[i] = self.samplers_list[i].__iter__()
                            cur_batch_sampler = self.sampler_iterators[i]
                            cur_sample_org = cur_batch_sampler.__next__()
                            cur_sample = cur_sample_org + push_index_val[i]
                            cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        logger.debug(f"Generated final_samples_list: {len(final_samples_list)}")
        self.final_samples_list = final_samples_list
        return final_samples_list

    def __len__(self):
        """Return number of samples"""
        # return len(self.dataset) * self.number_of_datasets
        # (idx, lang, sz) = self.dataset.get_datasets_info_by_desc_count()[0]
        # return ((sz + self.batch_size - 1) // self.batch_size) * self.number_of_datasets
        # return sz * self.number_of_datasets
        return len(self.final_samples_list)

    def __iter__(self):
        """Return iter of samples"""
        return iter(self.generate_sample_list())


def compose(*functions):
    # f ° g
    def compose2(g, f):
        return lambda x: g(f(x))

    return functools.reduce(compose2, functions, lambda x: x)


def build_lang_dataset(
    dirs: List[Path],
    name: str,
    data_params: DatasetParams,
    query_tokenizer: TokenizerRecordable,
    per_code_language_tokenizers: Dict[str, TokenizerRecordable],
    pickle_path=".",
    parallelize: bool = True,
) -> LangDataset:
    def build_input_features_from_dict(sample: Dict[str, Union[str, int, np.ndarray]]) -> InputFeatures:
        """Build InputFeature from Dict by randomizing between using docstring or function name for query"""
        if random.uniform(0.0, 1.0) < data_params.fraction_using_func_name:
            return InputFeatures(
                language=data_params.lang_ids[cast(str, sample["language"])],
                similarity=cast(int, sample["similarity"]),
                query_tokens=sample["query_tokens_func_name_as_query"],
                query_tokens_mask=sample["query_tokens_mask_func_name_as_query"],
                code_tokens=sample["code_tokens_func_name_as_query"],
                code_tokens_mask=sample["code_tokens_mask_func_name_as_query"],
            )
        else:
            return InputFeatures(
                language=data_params.lang_ids[cast(str, sample["language"])],
                similarity=cast(int, sample["similarity"]),
                query_tokens=sample["query_tokens_docstring_as_query"],
                query_tokens_mask=sample["query_tokens_mask_docstring_as_query"],
                code_tokens=sample["code_tokens_docstring_as_query"],
                code_tokens_mask=sample["code_tokens_mask_docstring_as_query"],
            )

    def parser(
        data_file: Path,
        data_params: DatasetParams,
        query_tokenizer: TokenizerRecordable,
        per_code_language_tokenizers: Dict[str, TokenizerRecordable],
    ) -> Tuple[str, int, Iterable[InputFeatures]]:
        (lang, lg, feats) = parse_data_file(data_file, data_params, query_tokenizer, per_code_language_tokenizers)
        return (lang, lg, list(map(build_input_features_from_dict, feats)))

    # Train Data
    if not os.path.exists(pickle_path):
        os.makedirs(pickle_path)

    pickle_file = Path(pickle_path) / f"{name}_samples.p"
    loaded_samples: Dict[str, Tuple[int, Iterable[InputFeatures]]]
    if os.path.exists(pickle_file):
        logger.debug(f"Loading dataset {name} raw samples from pickled {pickle_file}")
        loaded_samples = pickle.load(open(pickle_file, "rb"))
    else:
        logger.debug(f"Building dataset {name} from {dirs}")
        loaded_samples = load_data_from_dirs(
            data_dirs=dirs,
            query_tokenizer=query_tokenizer,
            per_code_language_tokenizers=per_code_language_tokenizers,
            data_params=data_params,
            parse_callback=parser,
            parallelize=parallelize,
        )
        nb = 0
        for lang, (lg, ss) in loaded_samples.items():
            ll = list(ss)
            loaded_samples[lang] = (lg, ll)
            nb += len(ll)
        pickle.dump(loaded_samples, open(pickle_file, "wb"))
        logger.debug(f"Pickled dataset {name} [{nb} raw samples] to {pickle_file}")

    # logger.debug(f"Samples {loaded_samples['python'][:2]}")
    dataset = LangDataset(loaded_samples, lang_ids=data_params.lang_ids)
    logger.debug(f"Loaded {name} lang dataset [{len(dataset)} samples]")
    return dataset


def build_lang_dataset_single_code_tokenizer(
    dirs: List[Path],
    name: str,
    data_params: DatasetParams,
    query_tokenizer: TokenizerRecordable,
    code_tokenizer: TokenizerRecordable,
    lang_token: str,
    pickle_path=".",
    parallelize: bool = False,
) -> LangDataset:
    def build_input_features_from_dict(sample: Dict[str, Union[str, int, np.ndarray]]) -> InputFeatures:
        """Build InputFeature from Dict by randomizing between using docstring or function name for query"""
        if random.uniform(0.0, 1.0) < data_params.fraction_using_func_name:
            return InputFeatures(
                language=data_params.lang_ids[cast(str, sample["language"])],
                similarity=cast(int, sample["similarity"]),
                query_tokens=sample["query_tokens_func_name_as_query"],
                query_tokens_mask=sample["query_tokens_mask_func_name_as_query"],
                code_tokens=sample["code_tokens_func_name_as_query"],
                code_tokens_mask=sample["code_tokens_mask_func_name_as_query"],
            )
        else:
            return InputFeatures(
                language=data_params.lang_ids[cast(str, sample["language"])],
                similarity=cast(int, sample["similarity"]),
                query_tokens=sample["query_tokens_docstring_as_query"],
                query_tokens_mask=sample["query_tokens_mask_docstring_as_query"],
                code_tokens=sample["code_tokens_docstring_as_query"],
                code_tokens_mask=sample["code_tokens_mask_docstring_as_query"],
            )

    def parser(
        data_file: Path,
        data_params: DatasetParams,
        query_tokenizer: TokenizerRecordable,
        code_tokenizer: TokenizerRecordable,
    ) -> Tuple[str, int, Iterable[InputFeatures]]:
        (lang, lg, feats) = parse_data_file_single_code_tokenizer(
            data_file, data_params, query_tokenizer, code_tokenizer, lang_token
        )
        return (lang, lg, list(map(build_input_features_from_dict, feats)))

    # Train Data
    if not os.path.exists(pickle_path):
        os.makedirs(pickle_path)

    pickle_file = Path(pickle_path) / f"{name}_samples.p"
    loaded_samples: Dict[str, Tuple[int, Iterable[InputFeatures]]]
    if os.path.exists(pickle_file):
        logger.debug(f"Loading dataset {name} raw samples from pickled {pickle_file}")
        loaded_samples = pickle.load(open(pickle_file, "rb"))
    else:
        logger.debug(f"Building dataset {name} from {dirs}")
        loaded_samples = load_data_from_dirs_single_code_tokenizer(
            data_dirs=dirs,
            query_tokenizer=query_tokenizer,
            code_tokenizer=code_tokenizer,
            data_params=data_params,
            parse_callback=parser,
            parallelize=parallelize,
        )
        nb = 0
        for lang, (lg, ss) in loaded_samples.items():
            ll = list(ss)
            loaded_samples[lang] = (lg, ll)
            nb += len(ll)
        pickle.dump(loaded_samples, open(pickle_file, "wb"))
        logger.debug(f"Pickled dataset {name} [{nb} raw samples] to {pickle_file}")

    # logger.debug(f"Samples {loaded_samples['python'][:2]}")
    dataset = LangDataset(loaded_samples, lang_ids=data_params.lang_ids)
    logger.debug(f"Loaded {name} lang dataset [{len(dataset)} samples]")
    return dataset


# class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
#     """


#     Sample elements randomly from a given list of indices for imbalanced dataset
#     Arguments:
#         indices (list, optional): a list of indices
#         num_samples (int, optional): number of samples to draw
#         callback_get_label func: a callback-like function which takes two arguments - dataset and index
#     """

#     def __init__(
#         self,
#         dataset: Dataset,
#         callback_get_label,  # function idx: Int -> str
#         label_to_count: Dict[str, int] = None,
#         indices=None,
#         num_samples=None,
#     ):

#         # if indices is not provided,
#         # all elements in the dataset will be considered
#         self.indices = list(range(len(dataset))) if indices is None else indices

#         # define custom callback
#         self.callback_get_label = callback_get_label

#         # if num_samples is not provided,
#         # draw `len(indices)` samples in each iteration
#         self.num_samples = len(self.indices) if num_samples is None else num_samples

#         # distribution of classes in the dataset
#         self.label_to_count: Dict[str, int] = {}
#         if label_to_count is not None:
#             self.label_to_count = label_to_count
#         else:
#             for idx in self.indices:
#                 label = self.callback_get_label(idx)
#                 if label in self.label_to_count:
#                     self.label_to_count[label] += 1
#                 else:
#                     self.label_to_count[label] = 1

#         # weight for each sample
#         weights: List[float] = [1.0 / self.label_to_count[self.callback_get_label(idx)] for idx in self.indices]
#         # print("weights", weights)
#         self.weights = torch.tensor(weights, dtype=torch.double)

#     def __iter__(self):
#         """Build iter"""
#         return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

#     def __len__(self):
#         """Return number of samples"""
#         return self.num_samples
