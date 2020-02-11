# Code partially copied and adapted from https://github.com/github/CodeSearchNet for backward-compatible experimentations

from collections import defaultdict

from typing import DefaultDict, Dict, Iterable, List, Optional, Tuple

from dpu_utils.mlutils import Vocabulary

from dpu_utils.utils import RichPath

from codenets.codesearchnet.copied_code.bpevocabulary import BpeVocabulary
from codenets.codesearchnet.copied_code.utils import run_jobs_in_parallel

from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from dataclasses import field
from enum import Enum

from codenets.utils import _to_subtoken_stream, get_data_files_from_directory


class QueryType(Enum):
    DOCSTRING = "docstring_as_query"
    FUNCTION_NAME = "func_name_as_query"


@dataclass
class Metadata:
    token_counter: Counter = field(default_factory=Counter)
    token_vocab: Optional[BpeVocabulary] = None
    common_tokens: List[Tuple[str, int]] = field(default_factory=list)


def load_metadata_from_sample(
    data_to_load: Iterable[str], raw_metadata: Metadata, use_subtokens: bool = False, mark_subtoken_end: bool = False
) -> Metadata:
    if use_subtokens:
        data_to_load = _to_subtoken_stream(data_to_load, mark_subtoken_end=mark_subtoken_end)
    # raw_metadata["token_counter"].update(data_to_load)
    raw_metadata.token_counter.update(data_to_load)
    return raw_metadata


def append_metadata(
    encoder_label: str,
    vocab_size: int,
    vocab_count_threshold: int,
    use_bpe: bool,
    pct_bpe: float,
    raw_metadata_list: List[Metadata],
) -> Metadata:
    merged_token_counter: Counter = Counter()
    for raw_metadata in raw_metadata_list:
        # merged_token_counter += raw_metadata["token_counter"]
        merged_token_counter += raw_metadata.token_counter

    # if hyperparameters["%s_use_bpe" % encoder_label]:
    token_vocabulary: Vocabulary
    if use_bpe:
        token_vocabulary = BpeVocabulary(
            # vocab_size=hyperparameters["%s_token_vocab_size" % encoder_label],
            vocab_size=vocab_size,
            # pct_bpe=hyperparameters["%s_pct_bpe" % encoder_label],
            pct_bpe=pct_bpe,
        )
        token_vocabulary.fit(merged_token_counter)
    else:
        token_vocabulary = Vocabulary.create_vocabulary(
            tokens=merged_token_counter,
            # max_size=hyperparameters["%s_token_vocab_size" % encoder_label],
            max_size=vocab_size,
            # count_threshold=hyperparameters["%s_token_vocab_count_threshold" % encoder_label],
            count_threshold=vocab_count_threshold,
        )

    # final_metadata["token_vocab"] = token_vocabulary
    # Save the most common tokens for use in data augmentation:
    # final_metadata["common_tokens"] = merged_token_counter.most_common(50)
    final_metadata = Metadata(
        token_vocab=token_vocabulary,
        token_counter=merged_token_counter,
        common_tokens=merged_token_counter.most_common(50),
    )
    return final_metadata


def build_tokenizer_metadata(
    data_dirs: List[Path],
    max_files_per_dir: Optional[int] = None,
    parallelize: bool = True,
    use_subtokens: bool = False,
    mark_subtoken_end: bool = False,
) -> Tuple[List[Metadata], Dict[str, List[Metadata]]]:
    raw_query_metadata_list = []
    raw_code_language_metadata_lists: DefaultDict[str, List] = defaultdict(list)

    def metadata_parser_fn(_, file_path: Path) -> Iterable[Tuple[Metadata, Dict[str, Metadata]]]:
        raw_query_metadata = Metadata()
        per_code_language_metadata: DefaultDict[str, Metadata] = defaultdict(Metadata)

        for raw_sample in RichPath.create(str(file_path)).read_by_file_suffix():
            sample_language = raw_sample["language"]
            per_code_language_metadata[sample_language] = load_metadata_from_sample(
                data_to_load=raw_sample["code_tokens"],
                raw_metadata=per_code_language_metadata[sample_language],
                use_subtokens=use_subtokens,
                mark_subtoken_end=mark_subtoken_end,
            )

            raw_query_metadata = load_metadata_from_sample(
                data_to_load=[d.lower() for d in raw_sample["docstring_tokens"]],
                raw_metadata=raw_query_metadata,
                use_subtokens=use_subtokens,
                mark_subtoken_end=mark_subtoken_end,
            )
        yield (raw_query_metadata, per_code_language_metadata)

    def received_result_callback(metadata_parser_result: Tuple[Metadata, Dict[str, Metadata]]):
        (raw_query_metadata, per_code_language_metadata) = metadata_parser_result
        raw_query_metadata_list.append(raw_query_metadata)
        for (metadata_language, raw_code_language_metadata) in per_code_language_metadata.items():
            raw_code_language_metadata_lists[metadata_language].append(raw_code_language_metadata)

    def finished_callback():
        pass

    if parallelize:
        run_jobs_in_parallel(
            get_data_files_from_directory(data_dirs, max_files_per_dir),
            metadata_parser_fn,
            received_result_callback,
            finished_callback,
        )
    else:
        for (idx, file) in enumerate(get_data_files_from_directory(data_dirs, max_files_per_dir)):
            for res in metadata_parser_fn(idx, file):
                received_result_callback(res)

    return raw_query_metadata_list, raw_code_language_metadata_lists
