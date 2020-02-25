from sentence_transformers import SentenceTransformer
from loguru import logger
import os
from codenets.codesearchnet.copied_code.utils import read_file_samples


"""Evaluating SBert."""


def main():
    from tree_sitter import Language, Parser

    # Language.build_library(
    #     # Store the library in the `build` directory
    #     "build/my-languages.so",
    #     # Include one or more languages
    #     [
    #         "vendor/tree-sitter-go",
    #         "vendor/tree-sitter-java",
    #         "vendor/tree-sitter-javascript",
    #         "vendor/tree-sitter-python",
    #         "vendor/tree-sitter-php",
    #         "vendor/tree-sitter-ruby",
    #     ],
    # )
    # model = SentenceTransformer("bert-base-nli-mean-tokens")
    # sentences = [
    #     "This framework generates embeddings for each input sentence",
    #     "Sentences are passed as a list of string.",
    #     "The quick brown fox jumps over the lazy dog.",
    # ]
    # sentence_embeddings = model.encode(sentences)
    # logger.info(f"sentence_embeddings {sentence_embeddings}")

    data_file = (
        "/home/mandubian/workspaces/tools/CodeSearchNet/resources/data/python/final/jsonl/valid/python_valid_0.jsonl.gz"
    )
    filename = os.path.basename(data_file)
    # file_language = filename.split("_")[0]

    samples = list(read_file_samples(data_file))

    logger.info(f"samples {samples[0]['code']}")
    logger.info(f"samples {samples[0]['code_tokens']}")

    PY_LANGUAGE = Language("build/my-languages.so", "python")
    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    tree = parser.parse(bytes(samples[0]["code"], "utf8"))

    logger.info(f"tree {tree}")


if __name__ == "__main__":
    main()
