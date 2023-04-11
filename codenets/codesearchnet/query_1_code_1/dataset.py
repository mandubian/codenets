import os
import sys
from typing import Iterable, Union, Dict, Tuple, List, Callable, TypeVar, Optional, Any, cast
import numpy as np
from pathlib import Path
from loguru import logger
from pathos.pools import ProcessPool
import itertools
import pickle
import random
from dpu_utils.codeutils import split_identifier_into_parts

from codenets.utils import _to_subtoken_stream, get_data_files_from_directory
from codenets.codesearchnet.data import DatasetParams
from codenets.codesearchnet.tokenizer_recs import TokenizerRecordable
from codenets.codesearchnet.copied_code.utils import read_file_samples
from codenets.codesearchnet.dataset_utils import (
    load_data_from_sample,
    Samples,
    LangDataset,
    Compose,
    InputFeaturesToNpArray,
    Tensorize,
)
from codenets.codesearchnet.copied_code.metadata import QueryType
from codenets.codesearchnet.data import InputFeatures


def convert_and_pad_token_sequence(
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
        tokens, tokens_mask = convert_and_pad_token_sequence(
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

    ds: List[Dict[str, Union[str, int, np.ndarray]]] = []
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


def load_data_from_files(
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
        per_file_results = [parse_callback(*task_args) for task_args in tasks_as_args]  # type: ignore[arg-type]

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
    return load_data_from_files(
        data_files=list(get_data_files_from_directory(data_dirs, max_files_per_dir)),
        data_params=data_params,
        query_tokenizer=query_tokenizer,
        code_tokenizer=code_tokenizer,
        parse_callback=parse_callback,
        parallelize=parallelize,
    )


def build_lang_dataset_single_code_tokenizer(
    dirs: List[Path],
    name: str,
    data_params: DatasetParams,
    query_tokenizer: TokenizerRecordable,
    code_tokenizer: TokenizerRecordable,
    lang_token: str,
    use_lang_weights: bool,
    pickle_path=".",
    parallelize: bool = False,
) -> LangDataset:
    # TODO : modify that because it should not be made at this step...
    # check query_code_siamese
    def build_input_features_from_dict(sample: Dict[str, Union[str, int, np.ndarray]]) -> InputFeatures:
        """Build InputFeature from Dict by randomizing between using docstring or function name for query"""
        if random.uniform(0.0, 1.0) < data_params.fraction_using_func_name:
            return InputFeatures(
                language=data_params.lang_ids[cast(str, sample["language"])],
                similarity=cast(int, sample["similarity"]),
                query_tokens=cast(np.ndarray, sample["query_tokens_func_name_as_query"]),
                query_tokens_mask=cast(np.ndarray, sample["query_tokens_mask_func_name_as_query"]),
                query_docstring_tokens=cast(np.ndarray, sample["query_tokens_docstring_as_query"]),
                query_docstring_tokens_mask=cast(np.ndarray, sample["query_tokens_mask_docstring_as_query"]),
                code_tokens=cast(np.ndarray, sample["code_tokens_func_name_as_query"]),
                code_tokens_mask=cast(np.ndarray, sample["code_tokens_mask_func_name_as_query"]),
            )
        else:
            return InputFeatures(
                language=data_params.lang_ids[cast(str, sample["language"])],
                similarity=cast(int, sample["similarity"]),
                query_tokens=cast(np.ndarray, sample["query_tokens_docstring_as_query"]),
                query_tokens_mask=cast(np.ndarray, sample["query_tokens_mask_docstring_as_query"]),
                query_docstring_tokens=cast(np.ndarray, sample["query_tokens_docstring_as_query"]),
                query_docstring_tokens_mask=cast(np.ndarray, sample["query_tokens_mask_docstring_as_query"]),
                code_tokens=cast(np.ndarray, sample["code_tokens_docstring_as_query"]),
                code_tokens_mask=cast(np.ndarray, sample["code_tokens_mask_docstring_as_query"]),
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
    transform = Compose([InputFeaturesToNpArray, Tensorize])
    dataset = LangDataset(
        loaded_samples, lang_ids=data_params.lang_ids,
        transform=transform, use_lang_weights=use_lang_weights,
        query_tokenizer=query_tokenizer,
        embedding_model=None
    )
    logger.debug(f"Loaded {name} lang dataset [{len(dataset)} samples]")
    return dataset
