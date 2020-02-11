import os
import sys
from typing import Iterable, Union, Dict, Tuple, List, Callable, TypeVar, Optional, cast
from pathlib import Path
from loguru import logger
from pathos.pools import ProcessPool
import itertools
import pickle
import random
import numpy as np

from codenets.utils import get_data_files_from_directory
from codenets.codesearchnet.data import DatasetParams
from codenets.codesearchnet.tokenizer_recs import TokenizerRecordable
from codenets.codesearchnet.original.utils import read_file_samples
from codenets.codesearchnet.dataset_utils import load_data_from_sample, Samples, LangDataset
from codenets.codesearchnet.data import InputFeatures


def parse_data_file(
    data_file: Path,
    data_params: DatasetParams,
    query_tokenizer: TokenizerRecordable,
    per_code_language_tokenizers: Dict[str, TokenizerRecordable],
) -> Tuple[str, int, Samples]:
    """
    Parse a file into a Tuple <file_language, len, Samples>
    Directly adapted from original Repo
    """

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


# The abstract type T returned by the parse_callback
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
    """
    Load data from many files using 

    Directly adapted from original repo
    """
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
