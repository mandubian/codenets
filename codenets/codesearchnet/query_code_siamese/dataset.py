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
from tqdm import tqdm
import pandas as pd
from dpu_utils.codeutils import split_identifier_into_parts

from codenets.utils import _to_subtoken_stream, get_data_files_from_directory
from codenets.codesearchnet.data import DatasetParams
from codenets.codesearchnet.tokenizer_recs import TokenizerRecordable
from codenets.codesearchnet.copied_code.utils import read_file_samples
from codenets.codesearchnet.dataset_utils import (
    Samples,
    LangDataset,
    LangDatasetDF,
    Compose,
    InputFeaturesToNpArray_RandomReplace,
    PDSeriesToNpArray,
    FullNpArrayToFinalNpArray,
    Tensorize,
    compute_language_weightings,
    compute_language_weightings_df,
)
from codenets.codesearchnet.copied_code.metadata import QueryType
from codenets.codesearchnet.data import InputFeatures
from codenets.codesearchnet.code_ast.ast_utils import load_special_tokens, TreeSitterParser


def convert_and_pad_token_sequence(
    tokenizer: TokenizerRecordable,
    token_sequence: List[str],
    output_tensor_size: int,
    token: str,
    prefix: Optional[str],
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
    if prefix is not None:
        token_sequence = [prefix, token] + token_sequence
    else:
        token_sequence = [token] + token_sequence
    token_ids, token_mask = tokenizer.encode_tokens([token_sequence], max_length=output_tensor_size)
    return token_ids[0], token_mask[0]


def load_data_from_sample_siamese(
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
    query_token: str,
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

        logger.debug("")
        if encoder_label == "code":
            tokens, tokens_mask = convert_and_pad_token_sequence(
                tokenizer=tokenizer,
                token_sequence=list(data),
                output_tensor_size=max_num_tokens,
                token=lang_token,
                prefix=language,
            )
        elif encoder_label == "query":
            tokens, tokens_mask = convert_and_pad_token_sequence(
                tokenizer=tokenizer,
                token_sequence=list(data),
                output_tensor_size=max_num_tokens,
                token=query_token,
                prefix=None,
            )
        # Note that we share the result_holder with different encoders, and so we need to make our identifiers
        # unique-ish
        result_holder[f"{encoder_label}_tokens_{key}"] = tokens
        result_holder[f"{encoder_label}_tokens_mask_{key}"] = tokens_mask

    if (
        result_holder[f"{encoder_label}_tokens_mask_{QueryType.DOCSTRING.value}"] is None
        or int(np.sum(result_holder[f"{encoder_label}_tokens_mask_{QueryType.DOCSTRING.value}"])) == 0
    ):
        return None

    return result_holder


def parse_data_file_siamese_tokenizer(
    data_file: Path, data_params: DatasetParams, tokenizer: TokenizerRecordable, lang_token: str, query_token: str
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
        data_code = load_data_from_sample_siamese(
            language=language,
            encoder_label="code",
            data_to_load=raw_sample["code_tokens"],
            function_name=function_name,
            tokenizer=tokenizer,
            fraction_using_func_name=data_params.fraction_using_func_name,
            min_len_func_name_for_query=data_params.min_len_func_name_for_query,
            use_subtokens=data_params.use_subtokens,
            mark_subtoken_end=data_params.mark_subtoken_end,
            max_num_tokens=data_params.code_max_num_tokens,
            lang_token=lang_token,
            query_token=query_token,
        )

        # query doesn't use the language
        data_query = load_data_from_sample_siamese(
            language=language,
            encoder_label="query",
            data_to_load=[d.lower() for d in raw_sample["docstring_tokens"]],
            function_name=function_name,
            tokenizer=tokenizer,
            fraction_using_func_name=data_params.fraction_using_func_name,
            min_len_func_name_for_query=data_params.min_len_func_name_for_query,
            use_subtokens=data_params.use_subtokens,
            mark_subtoken_end=data_params.mark_subtoken_end,
            max_num_tokens=data_params.query_max_num_tokens,
            lang_token=lang_token,
            query_token=query_token,
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
    tokenizer: TokenizerRecordable,
    # humm that is not very nice type signature... need to create interface for that
    parse_callback: Callable[[Path, DatasetParams, TokenizerRecordable], Tuple[str, int, Iterable[T_Single]]],
    parallelize: bool = True,
) -> Dict[str, Tuple[int, Iterable[T_Single]]]:
    tasks_as_args = [[data_file, data_params, tokenizer] for data_file in data_files]

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


def load_data_from_files_raw(
    data_files: Iterable[Path],
    # humm that is not very nice type signature... need to create interface for that
    parse_callback: Callable[..., Tuple[str, int, Iterable[T_Single]]],  # type: ignore
    parallelize: bool,
    *args,
) -> Dict[str, Tuple[int, Iterable[T_Single]]]:
    tasks_as_args = [[data_file, *args] for data_file in data_files]

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


def load_data_from_dirs_siamese_tokenizer(
    data_dirs: List[Path],
    tokenizer: TokenizerRecordable,
    data_params: DatasetParams,
    parse_callback: Callable[[Path, DatasetParams, TokenizerRecordable], Tuple[str, int, Iterable[T_Single]]],
    max_files_per_dir: Optional[int] = None,
    parallelize: bool = True,
) -> Dict[str, Tuple[int, Iterable[T_Single]]]:
    return load_data_from_files(
        data_files=list(get_data_files_from_directory(data_dirs, max_files_per_dir)),
        data_params=data_params,
        tokenizer=tokenizer,
        parse_callback=parse_callback,
        parallelize=parallelize,
    )


def load_data_from_dirs(
    data_dirs: List[Path],
    parse_callback: Callable[..., Tuple[str, int, Iterable[T_Single]]],  # type: ignore
    max_files_per_dir: Optional[int],
    parallelize: bool,
    *args,
) -> Dict[str, Tuple[int, Iterable[T_Single]]]:
    return load_data_from_files_raw(
        list(get_data_files_from_directory(data_dirs, max_files_per_dir)), parse_callback, parallelize, *args
    )


def build_lang_dataset_siamese_tokenizer(
    dirs: List[Path],
    name: str,
    data_params: DatasetParams,
    tokenizer: TokenizerRecordable,
    lang_token: str,
    query_token: str,
    fraction_using_func_name: float,
    query_random_token_frequency: float,
    common_tokens: Dict[int, List[int]],  # list of token ID
    use_lang_weights: bool,
    lang_ids: Dict[str, int],
    pickle_path=".",
    parallelize: bool = False,
    embedding_model=None,
) -> LangDataset:
    def build_input_features_from_dict(sample: Dict[str, Union[str, int, np.ndarray]]) -> InputFeatures:
        """Build InputFeature from Dict by randomizing between using docstring or function name for query"""
        return InputFeatures(
            language=data_params.lang_ids[cast(str, sample["language"])],
            similarity=cast(int, sample["similarity"]),
            query_tokens=sample["query_tokens_func_name_as_query"],
            query_tokens_mask=sample["query_tokens_mask_func_name_as_query"],
            query_docstring_tokens=sample["query_tokens_docstring_as_query"],
            query_docstring_tokens_mask=sample["query_tokens_mask_docstring_as_query"],
            code_tokens=sample["code_tokens_func_name_as_query"],
            code_tokens_mask=sample["code_tokens_mask_func_name_as_query"],
        )

    def parser(
        data_file: Path, data_params: DatasetParams, tokenizer: TokenizerRecordable
    ) -> Tuple[str, int, Iterable[InputFeatures]]:
        (lang, lg, feats) = parse_data_file_siamese_tokenizer(
            data_file, data_params, tokenizer, lang_token, query_token
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
        loaded_samples = load_data_from_dirs_siamese_tokenizer(
            data_dirs=dirs, tokenizer=tokenizer, data_params=data_params, parse_callback=parser, parallelize=parallelize
        )
        nb = 0
        for lang, (lg, ss) in loaded_samples.items():
            ll = list(ss)
            loaded_samples[lang] = (lg, ll)
            nb += len(ll)
        pickle.dump(loaded_samples, open(pickle_file, "wb"))
        logger.debug(f"Pickled dataset {name} [{nb} raw samples] to {pickle_file}")

    lang_weights = compute_language_weightings(loaded_samples, lang_ids)
    logger.debug(f"lang_weights {lang_weights}")

    transform = Compose(
        [
            InputFeaturesToNpArray_RandomReplace(
                lang_weights=lang_weights,
                fraction_using_func_name=fraction_using_func_name,
                query_random_token_frequency=query_random_token_frequency,
                common_tokens=common_tokens,
            ),
            Tensorize(),
        ]
    )
    dataset = LangDataset(
        loaded_samples,
        lang_ids=data_params.lang_ids,
        transform=transform,
        use_lang_weights=use_lang_weights,
        embedding_model=embedding_model,
        tokenizer=tokenizer,
        emb_annoy_path=Path(pickle_path) / f"{name}_embeddings.ann",
    )
    logger.debug(f"Loaded {name} lang dataset [{len(dataset)} samples]")
    return dataset


def load_data_from_sample_ast(
    language: str,
    encoder_label: str,
    data_to_load: List[str],
    function_name: Optional[str],
    tokenizer: TokenizerRecordable,
    data_params: DatasetParams,
    query_token: str,
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
    if (
        data_params.fraction_using_func_name > 0.0
        and function_name
        and len(function_name) >= data_params.min_len_func_name_for_query
    ):
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
        if data is not None:
            data_l: List[str] = list(data)
            if data_params.use_subtokens:
                data_l = list(_to_subtoken_stream(data_l, mark_subtoken_end=data_params.mark_subtoken_end))

            if encoder_label == "code":
                token_ids, token_mask = tokenizer.encode_tokens([data_l], max_length=data_params.code_max_num_tokens)

            elif encoder_label == "query":
                token_sequence = [query_token] + data_l
                token_ids, token_mask = tokenizer.encode_tokens(
                    [token_sequence], max_length=data_params.query_max_num_tokens
                )

            result_holder[f"{encoder_label}_tokens_{key}"] = token_ids[0]
            result_holder[f"{encoder_label}_tokens_mask_{key}"] = token_mask[0]

    if (
        result_holder[f"{encoder_label}_tokens_mask_{QueryType.DOCSTRING.value}"] is None
        or int(np.sum(result_holder[f"{encoder_label}_tokens_mask_{QueryType.DOCSTRING.value}"])) == 0
    ):
        return None

    return result_holder


def batch_iter(iterable, batch_size: int = 1):
    ln = len(iterable)
    for ndx in range(0, ln, batch_size):
        yield iterable[ndx : min(ndx + batch_size, ln)]


def parse_data_file_ast_tokenizer(
    data_file: Path,
    data_params: DatasetParams,
    tokenizer: TokenizerRecordable,
    ast_parser: TreeSitterParser,
    query_token: str,
    pickle_path: Path,
) -> Tuple[str, pd.DataFrame]:
    logger.info(f"Reading samples from {data_file}")
    filename = os.path.basename(data_file)
    file_language = filename.split("_")[0]
    file_id = filename.split(".")[0]
    pickle_file = pickle_path / f"{file_id}.p"

    if pickle_file.exists():
        df = pd.read_pickle(pickle_path / f"{file_id}.p")
        return (file_language, df)

    samples = list(read_file_samples(data_file))

    # ds: List[Dict[str, Union[str, int]]] = []
    codes: List[List[str]] = []
    funcs: List[List[str]] = []
    docstrings: List[List[str]] = []
    for idx, raw_sample in enumerate(tqdm(samples)):
        language = raw_sample["language"]
        if language.startswith("python"):  # In some datasets, we use 'python-2.7' and 'python-3'
            language = "python"

        if language != file_language:
            logger.error(f"file with different language {language} from filename {file_language}")
            sys.exit(f"file with multiple language {language} from filename {file_language}")

        function_name = raw_sample.get("func_name")

        code: List[str] = ast_parser.parse(language, raw_sample["code"], max_tokens=data_params.code_max_num_tokens)

        # Skip samples where the function name is very short, because it probably has too little information
        # to be a good search query.
        if (
            data_params.fraction_using_func_name > 0.0
            and function_name
            and len(function_name) >= data_params.min_len_func_name_for_query
        ):
            func = [query_token] + split_identifier_into_parts(function_name)
            code = [tokenizer.unk_token() if token == function_name else token for token in code]
            docstring = [query_token] + [d.lower() for d in raw_sample["docstring_tokens"]]

            codes.append(code)
            funcs.append(func)
            docstrings.append(docstring)

    code_toks: List[List[int]] = []
    code_masks: List[List[int]] = []
    func_toks: List[List[int]] = []
    func_masks: List[List[int]] = []
    docstring_toks: List[List[int]] = []
    docstring_masks: List[List[int]] = []

    for batch in batch_iter(codes, batch_size=100):
        toks, masks = tokenizer.encode_tokens(batch, max_length=data_params.code_max_num_tokens)
        code_toks.extend(toks)
        code_masks.extend(masks)

    for batch in batch_iter(funcs, batch_size=100):
        toks, masks = tokenizer.encode_tokens(batch, max_length=data_params.query_max_num_tokens)
        func_toks.extend(toks)
        func_masks.extend(masks)

    for batch in batch_iter(docstrings, batch_size=100):
        toks, masks = tokenizer.encode_tokens(batch, max_length=data_params.query_max_num_tokens)
        docstring_toks.extend(toks)
        docstring_masks.extend(masks)

    langs = [data_params.lang_ids[file_language]] * len(func_toks)
    similarities = [1] * len(func_toks)
    logger.debug(f"func_toks {func_toks[:2]}")
    logger.debug(f"docstring_toks {docstring_toks[:2]}")
    logger.debug(f"code_toks {code_toks[:2]}")
    logger.debug(f"langs {langs[:2]}")
    logger.debug(f"similarities {similarities[:2]}")
    df = pd.DataFrame(
        {
            "lang": langs,
            "similarity": similarities,
            "func_tokens": func_toks,
            "func_masks": func_masks,
            "docstring_tokens": docstring_toks,
            "docstring_masks": docstring_masks,
            "code_tokens": code_toks,
            "code_masks": code_masks,
        }
    )

    df.to_pickle(pickle_file)

    logger.debug(f"Saved file {data_file}: language {file_language} [{df.shape}] to {pickle_file}")

    return (file_language, df)


def load_data_from_files_ast(
    data_files: Iterable[Path],
    data_params: DatasetParams,
    tokenizer: TokenizerRecordable,
    ast_parser: TreeSitterParser,
    # humm that is not very nice type signature... need to create interface for that
    parse_callback: Callable[[Path, DatasetParams, TokenizerRecordable, TreeSitterParser], Tuple[str, pd.DataFrame]],
) -> Dict[str, pd.DataFrame]:
    tasks_as_args = [[data_file, data_params, tokenizer, ast_parser] for data_file in data_files]

    per_file_results = [parse_callback(*task_args) for task_args in tasks_as_args]  # type: ignore

    lang_samples_iter: Dict[str, List[pd.DataFrame]] = {}
    for (lang, df) in per_file_results:
        if lang not in lang_samples_iter:
            lang_samples_iter[lang] = []
        dfs = lang_samples_iter[lang]
        dfs.append(df)
        lang_samples_iter[lang] = dfs

    lang_samples: Dict[str, pd.DataFrame] = {}
    for (lang, dfs) in lang_samples_iter.items():
        lang_samples[lang] = pd.concat(dfs)

    return lang_samples


def load_data_from_dirs_ast(
    name: str,
    data_dirs: List[Path],
    tokenizer: TokenizerRecordable,
    ast_parser: TreeSitterParser,
    data_params: DatasetParams,
    parse_callback: Callable[[Path, DatasetParams, TokenizerRecordable, TreeSitterParser], Tuple[str, pd.DataFrame]],
) -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    for d in data_dirs:
        lg = os.path.basename(d.parents[2])
        logger.debug(f"Getting samples for lang {lg}")
        lang_samples = load_data_from_files_ast(
            data_files=list(get_data_files_from_directory([d], None)),
            data_params=data_params,
            tokenizer=tokenizer,
            ast_parser=ast_parser,
            parse_callback=parse_callback,
        )
        df = lang_samples[lg]

        logger.debug(f"lang {lg} ({df.shape[0]} samples)")

        dfs[lg] = df
    return dfs


def build_lang_dataset_ast(
    dirs: List[Path],
    name: str,
    data_params: DatasetParams,
    tokenizer: TokenizerRecordable,
    ast_parser: TreeSitterParser,
    query_token: str,
    common_tokens: Dict[int, List[int]],  # list of token ID
    pickle_path="./pickles",
) -> LangDatasetDF:
    def parser(
        data_file: Path, data_params: DatasetParams, tokenizer: TokenizerRecordable, parser: TreeSitterParser
    ) -> Tuple[str, pd.DataFrame]:
        (lang, df) = parse_data_file_ast_tokenizer(
            data_file, data_params, tokenizer, parser, query_token, pickle_path / name
        )
        return (lang, df)

    loaded_samples_dfs: Dict[str, pd.DataFrame]

    if not (pickle_path / name).exists():
        os.makedirs(pickle_path / name)

    ast_special_tokens = load_special_tokens(data_params)
    logger.debug(f"Adding special tokens {len(ast_special_tokens)} {ast_special_tokens} to tokenizer")
    tokenizer.add_special_tokens(ast_special_tokens)

    logger.debug(f"Building dataset {name} from {dirs}")
    loaded_samples_dfs = load_data_from_dirs_ast(
        name=name,
        data_dirs=dirs,
        tokenizer=tokenizer,
        ast_parser=ast_parser,
        data_params=data_params,
        parse_callback=parser,
    )

    lang_weights = compute_language_weightings_df(loaded_samples_dfs, data_params.lang_ids)
    logger.debug(f"lang_weights {lang_weights}")

    transform = Compose(
        [
            PDSeriesToNpArray(
                lang_weights=lang_weights,
                fraction_using_func_name=data_params.fraction_using_func_name,
                query_random_token_frequency=data_params.query_random_token_frequency,
                common_tokens=common_tokens,
            ),
            Tensorize(),
        ]
    )
    dataset = LangDatasetDF(
        loaded_samples_dfs,
        lang_ids=data_params.lang_ids,
        transform=transform,
        use_lang_weights=data_params.use_lang_weights,
        embedding_model=None,
        tokenizer=tokenizer,
        emb_annoy_path=Path(pickle_path) / f"{name}_embeddings.ann",
    )
    logger.debug(f"Loaded {name} lang dataset [{len(dataset)} samples]")
    return dataset
