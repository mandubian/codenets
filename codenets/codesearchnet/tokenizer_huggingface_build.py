#!/usr/bin/env python3
"""
Usage:
    tokenizers_huggingface_build.py [options]
    tokenizers_huggingface_build.py [options]

Options:
    -h --help                        Show this screen.
    --config FILE                    Specify HOCON config file. [default: ./conf/default.conf]
    --debug                          Enable debug routines. [default: False]
"""


from docopt import docopt
from loguru import logger
import sys
import torch
from typing import List
from dpu_utils.utils import run_and_debug
from pyhocon import ConfigFactory, ConfigTree
from codenets.codesearchnet.tokenizer_recs import (
    build_huggingface_bpetokenizers_from_hocon,
    build_huggingface_bpetokenizers_from_hocon_single_code_tokenizer,
    load_query_code_tokenizers_from_hocon,
    load_query_code_tokenizers_from_hocon_single_code_tokenizer,
    default_sample_update,
)


print("Torch version", torch.__version__)  # type: ignore

logger.remove()
logger.add(sys.stderr, level="DEBUG", colorize=True, backtrace=False)


def run(args, tag_in_vcs=False) -> None:
    conf_file = args["--config"]
    logger.info(f"config file {conf_file}")

    conf: ConfigTree = ConfigFactory.parse_file(conf_file)
    logger.info(f"config {conf}")

    def sample_update(tpe: str, lang: str, tokens: List[str]) -> str:
        if tpe == "code":
            return f"{lang} <lg> {' '.join(tokens)}\r\n"
        else:
            return default_sample_update(tpe, lang, tokens)

    build_huggingface_bpetokenizers_from_hocon(conf, from_dataset_type="train", sample_update=sample_update)

    tokenizers = load_query_code_tokenizers_from_hocon(conf)
    if tokenizers is not None:
        query_tokenizer, code_tokenizers = tokenizers

    txt = "python <lg> def toto():"
    toks = code_tokenizers["python"].tokenize(txt)
    enc = code_tokenizers["python"].encode_sentence(txt)
    logger.info(f"{txt} -> {toks}")
    logger.info(f"{txt} -> {enc}")

    # logger.info(f"encoded: {query_tokenizer.encode_sentence('Hello World')}")


def run_single_code_tokenizer(args, tag_in_vcs=False) -> None:
    conf_file = args["--config"]
    logger.info(f"config file {conf_file}")

    conf: ConfigTree = ConfigFactory.parse_file(conf_file)
    logger.info(f"config {conf}")

    def sample_update(tpe: str, lang: str, tokens: List[str]) -> str:
        if tpe == "code":
            return f"{lang} <lg> {' '.join(tokens)}\r\n"
        else:
            return default_sample_update(tpe, lang, tokens)

    build_huggingface_bpetokenizers_from_hocon_single_code_tokenizer(
        conf, from_dataset_type="train", sample_update=sample_update
    )

    tokenizers = load_query_code_tokenizers_from_hocon_single_code_tokenizer(conf)
    if tokenizers is not None:
        query_tokenizer, code_tokenizer = tokenizers

        txt = "python <lg> def toto():"
        toks = code_tokenizer.tokenize(txt)
        enc = code_tokenizer.encode_sentence(txt)
        logger.info(f"{txt} -> {toks}")
        logger.info(f"{txt} -> {enc}")

        txt = "go <lg> function getCounts() { return 0 }"
        toks = code_tokenizer.tokenize(txt)
        enc = code_tokenizer.encode_sentence(txt)
        logger.info(f"{txt} -> {toks}")
        logger.info(f"{txt} -> {enc}")
    else:
        logger.error("Couldn't load tokenizers")
    # logger.info(f"encoded: {query_tokenizer.encode_sentence('Hello World')}")


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run_single_code_tokenizer(args), args["--debug"])
