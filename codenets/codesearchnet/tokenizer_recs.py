from abc import ABC, abstractmethod

from typing import Iterable, List, Optional, Tuple, Union, Dict, cast, Callable
import numpy as np
from dpu_utils.mlutils import Vocabulary
from codenets.codesearchnet.original.bpevocabulary import BpeVocabulary
import os
from loguru import logger
from pathlib import Path
import pickle
from transformers import PreTrainedTokenizer, BertTokenizer

from tokenizers import BPETokenizer
from tokenizers.normalizers import BertNormalizer
from codenets.utils import get_data_files_from_directory, expand_data_path
from codenets.codesearchnet.original.utils import read_file_samples
from typing import IO
import time

from pyhocon import ConfigTree
from codenets.recordable import Recordable, instance_full_classname, full_classname, RecordableMapping
from codenets.codesearchnet.data import DatasetParams
from codenets.codesearchnet.original.metadata import Metadata, append_metadata, build_tokenizer_metadata


class TokenizerRecordable(ABC, Recordable):
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
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        pass

    @abstractmethod
    def encode_tokens(
        self, tokens: Iterable[List[str]], max_length: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        pass

    @abstractmethod
    def decode_sequence(self, tokens_sequence: List[int]) -> str:
        pass

    @abstractmethod
    def decode_sequences(self, tokens_sequences: Iterable[List[int]]) -> List[str]:
        pass


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


class BpeVocabularyTokenizerRecordable(TokenizerRecordable):
    def __init__(self, vocab: BpeVocabulary):
        self.vocab = vocab

    def save(self, output_dir: Union[Path, str]) -> bool:
        full_dir = Path(output_dir) / instance_full_classname(self)
        logger.debug(f"Saving BpeVocabularyTokenizerRecordable to {full_dir}")
        os.makedirs(full_dir, exist_ok=True)
        pickle.dump(self.vocab, open(full_dir / "vocab.pth", "wb"))
        return True

    @classmethod
    def load(cls, restore_dir: Union[Path, str]) -> "BpeVocabularyTokenizerRecordable":
        full_dir = Path(restore_dir) / full_classname(cls)
        logger.debug(f"Loading BpeVocabularyTokenizerRecordable from {full_dir}")
        vocab = pickle.load(open(full_dir / "vocab.pth", "rb"))
        return BpeVocabularyTokenizerRecordable(vocab)

    def tokenize(self, text: str, **kwargs) -> List[str]:
        return self.vocab.tokenize([text])

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return list(self.vocab.transform([tokens]))[0]

    def unk_token(self) -> str:
        return Vocabulary.get_unk()

    # def pad_token(self) -> str:
    #     return Vocabulary.get_pad()

    def encode_sentence(self, sentence: str, max_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        encoded = self.encode_sentences([sentence])
        return (encoded[0][0], encoded[1][0])

    def encode_sentences(
        self, sentences: List[str], max_length: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        tokens: List[List[str]] = [s.split(" ") for s in sentences]

        return self.encode_tokens(tokens, max_length)

    def encode_tokens(
        self, tokens: Iterable[List[str]], max_length: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if max_length is not None:
            token_idss = list(self.vocab.transform(tokens, fixed_length=max_length))
        else:
            token_idss = list(self.vocab.transform(tokens, fixed_length=None))
        # token_mask = np.array([1 if token_ids[i] > 0 else 0 for i in range(len(token_ids))])
        token_masks = []
        for i in range(len(token_idss)):
            token_masks.append([1 if token_idss[i][j] > 0 else 0 for j in range(len(token_idss[i]))])
        # np.array(
        #     [[1 if token_idss[i][j] > 0 else 0 for j in range(len(token_idss[i]))] for i in range(len(token_idss))]
        # )
        token_masks = [np.array(m) for m in token_masks]
        token_idss = [np.array(e) for e in token_idss]

        return token_idss, token_masks

    def decode_sequence(self, tokens_sequence: List[int]) -> str:
        return list(self.vocab.inverse_transform([tokens_sequence]))[0]

    def decode_sequences(self, tokens_sequences: Iterable[List[int]]) -> List[str]:
        return list(self.vocab.inverse_transform(tokens_sequences))


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


def build_original_tokenizers(
    data_dirs: List[Path],
    data_params: DatasetParams,
    max_files_per_dir: Optional[int] = None,
    parallelize: bool = True,
    default_tokenizers: Dict[str, TokenizerRecordable] = {},
) -> Tuple[TokenizerRecordable, Dict[str, TokenizerRecordable]]:

    start = time.time()

    query_metadata_lists, code_language_metadata_lists = build_tokenizer_metadata(
        data_dirs=data_dirs,
        max_files_per_dir=max_files_per_dir,
        parallelize=parallelize,
        use_subtokens=data_params.use_subtokens,
        mark_subtoken_end=data_params.mark_subtoken_end,
    )

    if len(query_metadata_lists) == 0:
        raise ValueError("Can't build tokenizers from empty metadata lists")

    query_tokenizer: TokenizerRecordable
    if "query" not in default_tokenizers:
        query_metadata = append_metadata(
            "query",
            vocab_size=data_params.vocab_size,
            vocab_count_threshold=data_params.vocab_count_threshold,
            use_bpe=data_params.use_bpe,
            pct_bpe=data_params.pct_bpe,
            raw_metadata_list=query_metadata_lists,
        )
        if query_metadata.token_vocab is not None:
            logger.info(f"using custom tokenizer for query")
            query_tokenizer = BpeVocabularyTokenizerRecordable(query_metadata.token_vocab)
    else:
        logger.info(f"using pretrained tokenizer for query")
        query_tokenizer = default_tokenizers["query"]

    # merge metadata if necessary
    per_code_language_metadata: Dict[str, Metadata] = {}
    for (language, raw_per_language_metadata) in code_language_metadata_lists.items():
        if language not in default_tokenizers:
            per_code_language_metadata[language] = append_metadata(
                "code",
                vocab_size=data_params.vocab_size,
                vocab_count_threshold=data_params.vocab_count_threshold,
                use_bpe=data_params.use_bpe,
                pct_bpe=data_params.pct_bpe,
                raw_metadata_list=raw_per_language_metadata,
            )

    per_code_language_tokenizer: Dict[str, TokenizerRecordable] = {}
    for language in per_code_language_metadata.keys():
        if language in default_tokenizers:
            logger.info(f"using pretrained tokenizer for language {language}")
            per_code_language_tokenizer[language] = default_tokenizers[language]
        else:
            logger.info(f"using custom tokenizer for language {language}")
            v: Optional[BpeVocabulary] = per_code_language_metadata[language].token_vocab
            if v is not None:
                per_code_language_tokenizer[language] = BpeVocabularyTokenizerRecordable(v)

    end = time.time()

    time_p = end - start
    logger.info(f"Full tokenizers training took: {time_p} sec")

    return query_tokenizer, per_code_language_tokenizer


def build_or_load_original_tokenizers(
    dirs: List[Path],
    name: str,
    data_params: DatasetParams,
    default_tokenizers: Dict[str, TokenizerRecordable] = {},
    pickle_path: str = ".",
    force_rebuild: bool = False,
) -> Tuple[TokenizerRecordable, Dict[str, TokenizerRecordable]]:
    if not os.path.exists(pickle_path):
        os.makedirs(pickle_path)
    pickle_file = Path(pickle_path) / f"{name}_tokenizers.p"
    query_tokenizer: TokenizerRecordable
    per_code_language_tokenizers: Dict[str, TokenizerRecordable]
    if os.path.exists(pickle_file) and not force_rebuild:
        logger.info(f"Loading tokenizer {name} from pickled {pickle_file}")
        query_tokenizer, per_code_language_tokenizers = pickle.load(open(pickle_file, "rb"))
    else:
        logger.info(f"Building tokenizer {name} from {dirs}")
        query_tokenizer, per_code_language_tokenizers = build_original_tokenizers(
            dirs, data_params, default_tokenizers={}
        )
        pickle.dump((query_tokenizer, per_code_language_tokenizers), open(pickle_file, "wb"))

    # testing query_tokenizer
    txt = "This is a docstring".lower()
    encoded_ids, encoded_mask = query_tokenizer.encode_sentence(txt)
    decoded = query_tokenizer.decode_sequence(encoded_ids)
    assert decoded == txt

    return query_tokenizer, per_code_language_tokenizers


def default_sample_update(tpe: str, lang: str, tokens: List[str]) -> str:
    return " ".join(tokens) + "\r\n"


def build_huggingface_token_files(
    data_dirs: List[Path],
    data_params: DatasetParams,
    output_path: Path,
    sample_update: Callable[[str, str, List[str]], str] = default_sample_update,
) -> Tuple[List[Path], Dict[str, Path]]:
    tokenizers_path = Path(output_path) / "huggingface_token_files"
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


def train_huggingface_bpetokenizers(
    data_params: DatasetParams, query_files: List[Path], lang_files: Dict[str, Path]
) -> Tuple[BPETokenizer, Dict[str, BPETokenizer]]:
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

    code_tokenizers = {}
    for lang, file_path in lang_files.items():
        logger.info(
            f"Building {lang} BPETokenizer from file {file_path} with do_lowercase:{data_params.do_lowercase} special_tokens:{data_params.special_tokens}"
        )
        code_tokenizers[lang] = BPETokenizer()
        code_tokenizers[lang].normalizer = BertNormalizer.new(
            clean_text=True, handle_chinese_chars=True, strip_accents=True, lowercase=data_params.do_lowercase
        )
        code_tokenizers[lang].train(
            files=str(file_path), vocab_size=data_params.vocab_size, special_tokens=data_params.special_tokens
        )

    return query_tokenizer, code_tokenizers


def train_huggingface_bpetokenizers_single_code_tokenizer(
    data_params: DatasetParams, query_files: List[Path], lang_files: Dict[str, Path]
) -> Tuple[BPETokenizer, Dict[str, BPETokenizer]]:
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

    return query_tokenizer, code_tokenizer


def build_huggingface_bpetokenizers(
    dirs: List[Path],
    data_params: DatasetParams,
    output_path: Union[str, Path] = ".",
    sample_update: Callable[[str, str, List[str]], str] = default_sample_update,
) -> Tuple[TokenizerRecordable, RecordableMapping]:
    output_path = Path(output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    start = time.time()

    query_files, lang_files = build_huggingface_token_files(dirs, data_params, output_path, sample_update)
    query_tokenizer, code_tokenizers = train_huggingface_bpetokenizers(data_params, query_files, lang_files)
    query_tokenizer_rec = HuggingfaceBPETokenizerRecordable(query_tokenizer)
    code_tokenizers_rec = RecordableMapping(
        {k: HuggingfaceBPETokenizerRecordable(t) for k, t in code_tokenizers.items()}
    )
    end = time.time()

    time_p = end - start
    logger.info(f"Full tokenizers training took: {time_p} sec")

    records = RecordableMapping({"query_tokenizer": query_tokenizer_rec, "code_tokenizers": code_tokenizers_rec})
    records.save(output_path)

    # testing query_tokenizer
    txt = "This is a docstring".lower()
    encoded_ids = query_tokenizer.encode(txt)
    logger.debug(f"encoded_ids {encoded_ids.tokens}")
    decoded = query_tokenizer.decode(encoded_ids.ids)
    logger.debug(f"decoded {decoded}")
    logger.debug(f"txt {txt}")
    # assert decoded == txt

    return query_tokenizer_rec, code_tokenizers_rec


def build_huggingface_bpetokenizers_single_code_tokenizer(
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
    query_tokenizer, code_tokenizer = train_huggingface_bpetokenizers_single_code_tokenizer(
        data_params, query_files, lang_files
    )
    query_tokenizer_rec = HuggingfaceBPETokenizerRecordable(query_tokenizer)
    code_tokenizer_rec = HuggingfaceBPETokenizerRecordable(code_tokenizer)
    end = time.time()

    time_p = end - start
    logger.info(f"query_tokenizer/code_tokenizer trainings took: {time_p} sec")

    records = RecordableMapping({"query_tokenizer": query_tokenizer_rec, "code_tokenizer": code_tokenizer_rec})
    records.save(output_path)

    # testing query_tokenizer
    txt = "This is a docstring".lower()
    encoded_ids = query_tokenizer.encode(txt)
    logger.debug(f"encoded_ids {encoded_ids.tokens}")
    decoded = query_tokenizer.decode(encoded_ids.ids)
    logger.debug(f"decoded {decoded}")
    logger.debug(f"txt {txt}")
    # assert decoded == txt

    return query_tokenizer_rec, code_tokenizer_rec


def build_huggingface_bpetokenizers_from_hocon(
    conf: ConfigTree,
    from_dataset_type="train",
    sample_update: Callable[[str, str, List[str]], str] = default_sample_update,
):
    dirs = expand_data_path(conf[f"dataset.{from_dataset_type}.dirs"])
    train_data_params = DatasetParams(**conf[f"dataset.{from_dataset_type}.params"])
    build_path = conf["tokenizers.build_path"]

    return build_huggingface_bpetokenizers(
        dirs=dirs, data_params=train_data_params, output_path=build_path, sample_update=sample_update
    )


def build_huggingface_bpetokenizers_from_hocon_single_code_tokenizer(
    conf: ConfigTree,
    from_dataset_type="train",
    sample_update: Callable[[str, str, List[str]], str] = default_sample_update,
) -> Tuple[TokenizerRecordable, TokenizerRecordable]:
    dirs = expand_data_path(conf[f"dataset.{from_dataset_type}.dirs"])
    train_data_params = DatasetParams(**conf[f"dataset.{from_dataset_type}.params"])
    build_path = conf["tokenizers.build_path"]

    return build_huggingface_bpetokenizers_single_code_tokenizer(
        dirs=dirs, data_params=train_data_params, output_path=build_path, sample_update=sample_update
    )


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


def load_query_code_tokenizers_from_hocon_single_code_tokenizer(
    conf: ConfigTree
) -> Optional[Tuple[TokenizerRecordable, TokenizerRecordable]]:
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
