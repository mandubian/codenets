# This file mixes code from original CodeSearchNet project ported to Pytorch and
# modified to consume less memory because it was exploding my 32GB RAM as provided
# on original code :(

from abc import abstractmethod

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Iterator, Callable
import re

import numpy as np
import random
import torch
import os
from pathlib import Path
from torch.utils.data import Dataset, TensorDataset, ConcatDataset, RandomSampler, Sampler
from torch import Tensor
from dpu_utils.codeutils import split_identifier_into_parts

from loguru import logger
import functools
import sklearn
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
import pandas as pd
from tqdm import tqdm
from enum import Enum

from codenets.codesearchnet.data import InputFeatures
from codenets.utils import _to_subtoken_stream
from codenets.codesearchnet.copied_code.metadata import QueryType
from codenets.codesearchnet.tokenizer_recs import TokenizerRecordable


class DatasetType(Enum):
    """Represents the 3 types of dataset (training, validation, test)"""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


# ugly type inherited from original CodeSearchNet code...
# will replace that later
Samples = Iterable[Dict[str, Union[str, int, np.ndarray]]]


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
) -> Optional[Dict[str, np.ndarray]]:
    """
    Save two versions of both the code and the query: one using the docstring as the query and the other using the
    function-name as the query, and replacing the function name in the code with an out-of-vocab token.
    Sub-tokenizes, converts, and pads both versions, and rejects empty samples.

    Adapted from original repo
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


class ConcatNamedDataset(Dataset):
    @abstractmethod
    def get_collate_fn(self) -> Optional[Callable[[List[Any]], List[Tensor]]]:
        return None

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


class FeatsDataset(Dataset):
    def __init__(self, samples: List[InputFeatures], transform: Callable[[InputFeatures, int], List[torch.Tensor]]):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        """Get dataset length"""
        return len(self.samples)

    def __getitem__(self, idx) -> List[torch.Tensor]:
        """Get dataset item by idx"""
        s = self.transform(self.samples[idx], idx)
        return s


class DFDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Callable[[pd.Series, int], List[torch.Tensor]]):
        self.df = df
        self.transform = transform

    def __len__(self):
        """Get dataset length"""
        return self.df.shape[0]
        # return len(self.values)

    def __getitem__(self, idx) -> List[torch.Tensor]:
        """Get dataset item by idx"""
        s = self.transform(self.df.iloc[idx], idx)
        return s


class InputFeaturesToNpArray(object):
    def __init__(self, lang_weights: Optional[Dict[int, float]]):
        self.lang_weights = lang_weights

    def __call_(self, features: Iterable[InputFeatures]):
        all_query_tokens = [f.query_tokens for f in features]
        all_query_tokens_mask = [f.query_tokens_mask for f in features]
        all_code_tokens = [f.code_tokens for f in features]
        all_code_tokens_mask = [f.code_tokens_mask for f in features]
        all_languages = [f.language for f in features]
        all_similarity = [f.similarity for f in features]
        if self.lang_weights is not None:
            all_lang_weights = [self.lang_weights[f.language] for f in features]
        else:
            all_lang_weights = [1.0 for _ in features]
        return (
            all_languages,
            all_similarity,
            all_query_tokens,
            all_query_tokens_mask,
            all_code_tokens,
            all_code_tokens_mask,
            all_lang_weights,
        )


class InputFeaturesToNpArray_RandomReplace(object):
    def __init__(
        self,
        lang_weights: Optional[Dict[int, float]],
        fraction_using_func_name: float,
        query_random_token_frequency: float,
        common_tokens: Dict[int, List[int]],
    ):
        self.lang_weights = lang_weights
        self.fraction_using_func_name = fraction_using_func_name
        self.query_random_token_frequency = query_random_token_frequency
        self.common_tokens = common_tokens

    def __call__(self, feat: InputFeatures, idx: int) -> List[np.ndarray]:

        if random.uniform(0.0, 1.0) < self.fraction_using_func_name:
            query_tokens = feat.query_tokens
            query_tokens_mask = feat.query_tokens_mask
        else:
            query_tokens = feat.query_docstring_tokens
            query_tokens_mask = feat.query_docstring_tokens_mask

        code_tokens = feat.code_tokens
        code_tokens_mask = feat.code_tokens_mask
        language = feat.language
        similarity = feat.similarity
        if self.lang_weights is not None:
            lang_weights = self.lang_weights[feat.language]
        else:
            lang_weights = 1.0

        # only replace tokens with most common tokens if there are
        if language in self.common_tokens and len(self.common_tokens[language]) > 0:
            total_length = len(query_tokens)
            length_without_padding = int(np.sum(query_tokens_mask))
            insert_indices = np.array([random.uniform(0.0, 1.0) for _ in range(length_without_padding)])
            insert_indices = insert_indices < self.query_random_token_frequency
            insert_indices = np.flatnonzero(insert_indices)
            if len(insert_indices) > 0:
                tokens_to_add = [
                    random.randrange(0, len(self.common_tokens[language])) for _ in range(len(insert_indices))
                ]  # select one of the most common tokens for each location
                tokens_to_add = [
                    self.common_tokens[language][token] for token in tokens_to_add
                ]  # get the ID corresponding to the token we're adding
                to_insert = 0
                output_query = np.zeros(total_length, dtype=int)
                for idx in range(
                    min(length_without_padding, total_length - len(insert_indices))
                ):  # iterate only through the beginning of the array where changes are being made
                    if to_insert < len(insert_indices) and idx == insert_indices[to_insert]:
                        output_query[idx + to_insert] = tokens_to_add[to_insert]
                        to_insert += 1
                    output_query[idx + to_insert] = query_tokens[idx]
                query_tokens = output_query
                # Add the needed number of non-padding values to the mask:
                query_tokens_mask[length_without_padding : length_without_padding + len(tokens_to_add)] = 1.0

        return [idx, language, similarity, query_tokens, query_tokens_mask, code_tokens, code_tokens_mask, lang_weights]


class FullNpArrayToFinalNpArray(object):
    def __init__(
        self,
        lang_weights: Optional[Dict[int, float]],
        fraction_using_func_name: float,
        query_random_token_frequency: float,
        common_tokens: Dict[int, List[int]],
    ):
        self.lang_weights = lang_weights
        self.fraction_using_func_name = fraction_using_func_name
        self.query_random_token_frequency = query_random_token_frequency
        self.common_tokens = common_tokens

    def __call__(self, feat: pd.Series, idx: int) -> List[np.ndarray]:

        if random.uniform(0.0, 1.0) < self.fraction_using_func_name:
            query_tokens = feat[2]
            query_tokens_mask = feat[3]
        else:
            query_tokens = feat[4]
            query_tokens_mask = feat[5]

        code_tokens = feat[6]
        code_tokens_mask = feat[7]

        language = feat[0]
        similarity = feat[1]
        if self.lang_weights is not None:
            lang_weights = self.lang_weights[language]
        else:
            lang_weights = 1.0

        return [idx, language, similarity, query_tokens, query_tokens_mask, code_tokens, code_tokens_mask, lang_weights]


class PDSeriesToNpArray(object):
    def __init__(
        self,
        lang_weights: Optional[Dict[int, float]],
        fraction_using_func_name: float,
        query_random_token_frequency: float,
        common_tokens: Dict[int, List[int]],
    ):
        self.lang_weights = lang_weights
        self.fraction_using_func_name = fraction_using_func_name
        self.query_random_token_frequency = query_random_token_frequency
        self.common_tokens = common_tokens

    def __call__(self, feat: pd.Series, idx: int) -> List[np.ndarray]:

        if random.uniform(0.0, 1.0) < self.fraction_using_func_name:
            query_tokens = feat["func_tokens"]
            query_tokens_mask = feat["func_masks"]
        else:
            query_tokens = feat["docstring_tokens"]
            query_tokens_mask = feat["docstring_masks"]

        code_tokens = feat["code_tokens"]
        code_tokens_mask = feat["code_masks"]

        language = feat["lang"]
        similarity = feat["similarity"]
        if self.lang_weights is not None:
            lang_weights = self.lang_weights[language]
        else:
            lang_weights = 1.0

        return [idx, language, similarity, query_tokens, query_tokens_mask, code_tokens, code_tokens_mask, lang_weights]


class Tensorize(object):
    def __call__(self, features: Iterable[np.ndarray]) -> List[torch.Tensor]:
        res = [torch.as_tensor(f) for f in features]
        return res


class Compose(object):
    """
    Compose several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        if len(self.transforms) > 0:
            d = self.transforms[0](*args)
            for t in self.transforms[1:]:
                d = t(d)
        return d

    def __repr__(self):
        """Represent object"""
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class LangDataset(ConcatNamedDataset):
    """
    Dataset wrapping language features.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(
        self,
        lang_features: Dict[str, Tuple[int, Iterable[InputFeatures]]],
        lang_ids: Dict[str, int],
        transform: Callable[[InputFeatures, int], List[np.ndarray]],
        use_lang_weights: bool = False,
        # query_embeddings: Optional[Dict[str, Tuple[int, Iterable[np.ndarray]]]] = None,
        embedding_model=None,
        tokenizer=None,
        emb_annoy_path: Path = None,
    ):
        super(LangDataset, self).__init__()

        self.langs: List[str] = list(lang_features.keys())
        self.lang_ids = lang_ids
        # self.datasets: List[TensorDataset] = []
        self.datasets: List[FeatsDataset] = []

        self.datasets_len: List[Tuple[int, str, int]] = []
        self.lang_indexes: Dict[int, int] = {}
        self.use_lang_weights = use_lang_weights

        def filter_features(samples: List[InputFeatures]) -> List[InputFeatures]:
            full_bads = 0
            toks_2_docs = 0
            docs_2_toks = 0
            res: List[InputFeatures] = []
            for s in samples:
                toks_len = len(s.query_tokens_mask[s.query_tokens_mask != 0])
                toks = s.query_tokens[: len(s.query_tokens_mask[s.query_tokens_mask != 0])]
                toks = tokenizer.decode_sequence(toks)
                toks_1 = re.sub(r"[^a-zA-Z0-9\s]+", "", toks[5:]).strip()
                docs_len = len(s.query_docstring_tokens_mask[s.query_docstring_tokens_mask != 0])
                docs = s.query_docstring_tokens[:docs_len]
                docs = tokenizer.decode_sequence(docs)
                docs_1 = re.sub(r"[^a-zA-Z0-9\s]+", "", docs[5:]).strip()

                if toks_len < 3 or len(toks_1) == 0:
                    bad_tok = True
                else:
                    bad_tok = False

                if docs_len < 2 or len(docs_1) == 0:
                    bad_doc = True
                else:
                    bad_doc = False

                if not bad_tok and not bad_doc:
                    res.append(s)
                elif not bad_tok and bad_doc:
                    # put tok in doc to force having at least one correct field
                    toks_2_docs += 1
                    s.query_docstring_tokens = s.query_tokens
                    s.query_docstring_tokens_mask = s.query_tokens_mask
                    res.append(s)
                elif bad_tok and not bad_doc:
                    docs_2_toks += 1
                    # put doc in tok to force having at least one correct field
                    s.query_tokens = s.query_docstring_tokens
                    s.query_tokens_mask = s.query_docstring_tokens_mask
                    res.append(s)
                else:
                    # both bad, skip sample
                    full_bads += 1

            logger.debug(
                f"Samples before:{len(samples)} after:{len(res)} full_bads:{full_bads} toks_2_docs:{toks_2_docs} docs_2_toks:{docs_2_toks}"
            )
            return res

        logger.info("Concatenating Datasets")
        lang_features_sorted = sorted(lang_features.items(), key=lambda lf: lf[1][0], reverse=True)
        for idx, (lang, (nb, features)) in enumerate(lang_features_sorted):
            lang_id = self.lang_ids[lang]
            self.lang_indexes[lang_id] = idx
            logger.info(f"Adding Language {lang} id:{idx} lang_id:{lang_id} [{nb} samples]")
            self.datasets_len.append((lang_id, lang, nb))
            fs = list(features)
            ds = FeatsDataset(filter_features(fs), transform)
            self.datasets.append(ds)

        self.concat_dataset: ConcatDataset = ConcatDataset(self.datasets)
        logger.info(f"Concat_dataset [{len(self.concat_dataset)} samples]")

        all_embs_df: pd.DataFrame
        if embedding_model is not None:
            from annoy import AnnoyIndex

            annoy_index = AnnoyIndex(768, "angular")
            if emb_annoy_path is not None and emb_annoy_path.exists():
                logger.debug(f"Loading Embeddings from AnnoyIndex {emb_annoy_path}")
                # all_embs_df = pd.read_parquet(emb_parquet_path, columns=["embeddings"])

                annoy_index.load(str(emb_annoy_path))

            else:
                logger.debug(f"Building Embeddings to AnnoyIndex {emb_annoy_path}")
                for idx in tqdm(range(0, len(self.concat_dataset))):
                    sample = self.concat_dataset[idx]
                    ts = sample[3]
                    mask = sample[4]
                    toks = ts.cpu().numpy()[: len(mask[mask != 0])]
                    s = tokenizer.decode_sequence(toks)
                    s = re.sub(r"[^a-zA-Z0-9\s]+", "", s[5:]).strip()
                    embs = embedding_model.encode([s])
                    annoy_index.add_item(idx, embs[0])
                annoy_index.build(10)  # 10 trees
                annoy_index.save(str(emb_annoy_path))

                # samples = [
                #     self.concat_dataset[i]
                #     for i in range(
                #         idx,
                #         idx + batch_size
                #         if idx + batch_size < len(self.concat_dataset)
                #         else len(self.concat_dataset),
                #     )
                # ]
                # query_tokens_masks = [(sample[3], sample[4]) for sample in samples]
                # toks = [ts.cpu().numpy()[: len(mask[mask != 0])] for (ts, mask) in query_tokens_masks]
                # toks = tokenizer.decode_sequences(toks)
                # # removes "<qy> "
                # toks = [re.sub(r"[^a-zA-Z0-9\s]+", "", t[5:]).strip() for t in toks]
                # # logger.debug(f"toks {toks}")
                # embs = embedding_model.encode(toks)
                # all_embs.extend([[emb] for emb in embs])

                # all_embs_df = pd.DataFrame(all_embs, columns=["embeddings"])
                # logger.debug(f"all_embs_df {all_embs_df.info()}")
                # all_embs_df.to_parquet(emb_parquet_path)
                logger.debug(f"Saved Embedding AnnoyIndex to {emb_annoy_path}")

                # free model from mem
                del embedding_model

            # logger.debug(f"all_embs_df {all_embs_df.info()}")
            # self.query_embeddings = query_embeddings
            collate_fn: Optional[Callable[[List[Any]], List[Tensor]]]
            # if self.query_embeddings is not None:

            # ensuring we have the same order in embeddings as the datasets above
            # all_embs: List[np.ndarray] = []
            # for _, (lang, (_, _)) in enumerate(lang_features_sorted):
            #     nb, embs = self.query_embeddings[lang]
            #     all_embs.extend([[emb] for emb in embs])

            # all_embs_df = pd.DataFrame(all_embs)

            def collate_fn_emb(batch):
                all_tensors = [torch.stack([row[i] for row in batch]) for i in range(len(batch[0]))]
                indices = all_tensors[0]
                # logger.debug(f"indices {indices}")
                similarities = torch.zeros(len(indices), len(indices))
                for idx, i in enumerate(indices):
                    dists = [1.0 - annoy_index.get_distance(i, j) for j in indices]
                    similarities[idx, :] = torch.tensor(dists)
                    similarities[:, idx] = torch.tensor(dists)
                # similarities = torch.tensor([[1.0 - annoy_index.get_distance(i, j) for j in indices] for i in indices])
                # batch_embs = np.stack(np.squeeze(all_embs_df.iloc[indices].values))
                margin = 0.25
                # similarities = (torch.tensor(cosine_similarity(batch_embs, batch_embs)) - margin).clamp(min=0.0)
                similarities = (similarities - margin).clamp(min=0.0)
                similarities.fill_diagonal_(1.0)
                # logger.debug(f"similarities {similarities}")
                all_tensors[2] = similarities

                return all_tensors

            collate_fn = collate_fn_emb
        else:
            collate_fn = None

        self.collate_fn = collate_fn

    def get_collate_fn(self) -> Optional[Callable[[List[Any]], List[Tensor]]]:
        return self.collate_fn

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


class LangDatasetDF(ConcatNamedDataset):
    """
    Dataset wrapping language features.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(
        self,
        lang_features_df: Dict[str, pd.DataFrame],
        lang_ids: Dict[str, int],
        transform: Callable[[pd.Series, int], List[np.ndarray]],
        use_lang_weights: bool = False,
        # query_embeddings: Optional[Dict[str, Tuple[int, Iterable[np.ndarray]]]] = None,
        embedding_model=None,
        tokenizer=None,
        emb_annoy_path: Path = None,
    ):
        super(LangDatasetDF, self).__init__()

        self.langs: List[str] = list(lang_features_df.keys())
        self.lang_ids = lang_ids
        # self.datasets: List[TensorDataset] = []
        self.datasets: List[DFDataset] = []

        self.datasets_len: List[Tuple[int, str, int]] = []
        self.lang_indexes: Dict[int, int] = {}
        self.use_lang_weights = use_lang_weights

        logger.info("Concatenating Datasets")
        lang_features_sorted = sorted(lang_features_df.items(), key=lambda lf: lf[1].shape[0], reverse=True)
        for idx, (lang, df) in enumerate(lang_features_sorted):
            lang_id = self.lang_ids[lang]
            self.lang_indexes[lang_id] = idx
            logger.info(f"Adding Language {lang} id:{idx} lang_id:{lang_id} [{df.shape[0]} samples]")
            self.datasets_len.append((lang_id, lang, df.shape[0]))
            # fs = list(features)
            # ds = FeatsDataset(filter_features(fs), transform)
            ds = DFDataset(df, transform)
            self.datasets.append(ds)

        self.concat_dataset: ConcatDataset = ConcatDataset(self.datasets)
        logger.info(f"Concat_dataset [{len(self.concat_dataset)} samples]")

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

    def get_collate_fn(self) -> Optional[Callable[[List[Any]], List[Tensor]]]:
        return None

    def __getitem__(self, idx: int):
        """Get item"""
        return self.concat_dataset[idx]

    def __len__(self):
        """Get length"""
        return len(self.concat_dataset)


def compute_language_weightings(
    data: Dict[str, Tuple[int, Iterable[InputFeatures]]], lang_ids: Dict[str, int]
) -> Dict[int, float]:
    # language_to_num_remaining_samples = {}
    # for (language, (nb, _)) in data.items():
    #     language_to_num_remaining_samples[language] = nb

    total_num_samples = sum(map(lambda d: d[0], data.values()))
    num_languages = len(data)
    language_to_reweighting_factor = {
        lang_ids[language]: float(total_num_samples) / (num_languages * nb) for (language, (nb, _)) in data.items()
    }

    return language_to_reweighting_factor


def compute_language_weightings_df(data: Dict[str, pd.DataFrame], lang_ids: Dict[str, int]) -> Dict[int, float]:
    total_num_samples = 0
    language_to_reweighting_factor: Dict[int, float] = {}
    num_languages = len(data)
    for (language, df) in data.items():
        ln = df.shape[0]
        language_to_reweighting_factor[lang_ids[language]] = 1 / (num_languages * ln)
        total_num_samples += ln

    language_to_reweighting_factor = {
        lid: total_num_samples * factor for lid, factor in language_to_reweighting_factor.items()
    }

    return language_to_reweighting_factor


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
