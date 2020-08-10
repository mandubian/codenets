from loguru import logger
import time
from typing import Dict, List, Tuple, IO, Set, Optional
from pathlib import Path
from tree_sitter import Language, Parser, Node
import os
import json
from codenets.codesearchnet.data import DatasetParams
from codenets.utils import get_data_files_from_directory
from codenets.codesearchnet.copied_code.utils import read_file_samples


class TreeSitterParser:
    def __init__(
        self,
        langs: List[str],
        added_nodes: Dict[str, Dict[str, str]],
        skip_node_types: Dict[str, List[str]],
        vendors_path: Path = Path("./vendor"),
    ):
        super(TreeSitterParser, self).__init__()

        vendors = []
        self.added_nodes = added_nodes
        self.skip_node_types = skip_node_types
        for lang in langs:
            vendors.append(vendors_path / f"tree-sitter-{lang}")
            if lang not in added_nodes:
                self.added_nodes[lang] = {"prefix": "", "suffix": ""}
            if lang not in skip_node_types:
                self.skip_node_types[lang] = []

        Language.build_library(
            # Store the library in the `build` directory
            "build/my-languages.so",
            # Include one or more languages
            vendors,
        )

        self.parser = Parser()

    def repr_field_node(
        self, code, node, field: Optional[str] = None, skip_node_types: List[str] = []
    ) -> Tuple[List[str], Set[str], bool]:
        skip_sub_nodes = False
        special_tokens: Set[str] = set()
        rpr: List[str]
        if field:
            rpr = ["<field>", field]
            special_tokens.add("<field>")
        else:
            rpr = []

        if node.is_named:
            # no child, serialize it here
            if len(node.children) == 0:
                if node.type in skip_node_types:
                    rpr.extend([f"{node.type}", "<nosub>"])
                    special_tokens.add("<nosub>")
                else:
                    rpr.extend([f"<{node.type}>", code[node.start_byte : node.end_byte], "<nosub>"])
                    special_tokens.update([f"<{node.type}>", "<nosub>"])

            else:
                if node.type not in skip_node_types:
                    rpr.extend([f"<{node.type}>", "<sub>"])
                    special_tokens.update([f"<{node.type}>", "<sub>"])
                else:
                    skip_sub_nodes = True
        else:
            if node.type not in skip_node_types:
                rpr.extend([f"{node.type}", "<nosub>"])
                special_tokens.add("<nosub>")
            else:
                skip_sub_nodes = True

        return rpr, special_tokens, skip_sub_nodes

    def repr_level(self, code, cursor, skip_node_types: List[str] = []):
        nodes: List[Node] = []
        all_tokens: List[str] = []
        special_tokens: Set[str] = set()

        if cursor.goto_first_child():
            field = cursor.current_field_name()
            toks, specs, skip = self.repr_field_node(code, cursor.node, field, skip_node_types=skip_node_types)
            all_tokens.extend(toks)
            special_tokens.update(specs)
            if not skip:
                nodes.append(cursor.node)

            while cursor.goto_next_sibling():
                field = cursor.current_field_name()
                toks, specs, skip = self.repr_field_node(code, cursor.node, field, skip_node_types=skip_node_types)
                all_tokens.extend(toks)
                special_tokens.update(specs)
                if not skip:
                    nodes.append(cursor.node)

            all_tokens.append("<lvl>")
            special_tokens.add("<lvl>")
        return all_tokens, special_tokens, nodes

    def breadth_first_path(self, lang, code, cursor, skip_node_types: List[str] = []) -> Tuple[List[str], Set[str]]:
        all_tokens = [f"<{lang}>"]
        special_tokens = set([f"<{lang}>"])
        all_tokens_1, special_tokens_1, skip = self.repr_field_node(code, cursor.node, skip_node_types=skip_node_types)
        all_tokens.extend(all_tokens_1)
        special_tokens.update(special_tokens_1)

        if not skip:
            all_tokens_lvl, special_tokens_lvl, nodes = self.repr_level(code, cursor, skip_node_types=skip_node_types)
            all_tokens.extend(all_tokens_lvl)
            special_tokens.update(special_tokens_lvl)

            while len(nodes) > 0:
                node = nodes.pop(0)
                cursor = node.walk()
                all_tokens_2, special_tokens_2, nodes_2 = self.repr_level(code, cursor, skip_node_types=skip_node_types)
                all_tokens.extend(all_tokens_2)
                special_tokens.update(special_tokens_2)
                nodes.extend(nodes_2)
        all_tokens.append("<end>")
        special_tokens.add("<end>")
        return all_tokens, special_tokens

    def breadth_first_path_light(
        self, lang, code, cursor, skip_node_types: List[str] = [], max_tokens: Optional[int] = None
    ) -> List[str]:
        all_tokens = [f"<{lang}>"]
        all_tokens_1, special_tokens_1, skip = self.repr_field_node(code, cursor.node, skip_node_types=skip_node_types)
        all_tokens.extend(all_tokens_1)

        if not skip:
            all_tokens_lvl, special_tokens_lvl, nodes = self.repr_level(code, cursor, skip_node_types=skip_node_types)
            all_tokens.extend(all_tokens_lvl)

            while len(nodes) > 0 and len(all_tokens) < max_tokens:
                node = nodes.pop(0)
                cursor = node.walk()
                all_tokens_2, special_tokens_2, nodes_2 = self.repr_level(code, cursor, skip_node_types=skip_node_types)
                all_tokens.extend(all_tokens_2)
                nodes.extend(nodes_2)
        all_tokens = all_tokens[:max_tokens]
        all_tokens.append("<end>")
        return all_tokens

    def parse_full(self, lang: str, code: str) -> Tuple[List[str], Set[str]]:
        LANGUAGE = Language("build/my-languages.so", lang)
        self.parser.set_language(LANGUAGE)

        code = f"{self.added_nodes[lang]['prefix']} {code} {self.added_nodes[lang]['suffix']}"

        tree = self.parser.parse(bytes(code, "utf8"))
        cursor = tree.walk()

        tokens, special_tokens = self.breadth_first_path(lang, code, cursor, skip_node_types=self.skip_node_types[lang])
        return tokens, special_tokens

    def parse(self, lang: str, code: str, max_tokens: Optional[int] = None) -> List[str]:
        LANGUAGE = Language("build/my-languages.so", lang)
        self.parser.set_language(LANGUAGE)

        code = f"{self.added_nodes[lang]['prefix']} {code} {self.added_nodes[lang]['suffix']}"

        tree = self.parser.parse(bytes(code, "utf8"))
        cursor = tree.walk()

        tokens = self.breadth_first_path_light(
            lang, code, cursor, skip_node_types=self.skip_node_types[lang], max_tokens=max_tokens
        )
        return tokens


def load_special_tokens(data_params: DatasetParams):
    special_tokens: List[str] = []
    for f in data_params.ast_special_tokens_files:
        fp = open(f, "r")
        special_tokens.extend(json.load(fp))

    return special_tokens


def build_language_ast(name: str, dirs: List[Path], pickle_path: Path, data_params: DatasetParams):
    start = time.time()

    if data_params.use_ast == "tree-sitter":
        parser = TreeSitterParser(
            langs=["go", "java", "javascript", "python", "php", "ruby"],
            added_nodes=data_params.ast_added_nodes,
            skip_node_types=data_params.ast_skip_node_types,
        )

        all_special_tokens: Set[str] = set()

        lengths: Dict[str, List[int]] = {"go": [], "java": [], "javascript": [], "python": [], "php": [], "ruby": []}

        for (idx, file_path) in enumerate(get_data_files_from_directory(dirs)):
            logger.info(f"Reading {file_path}")
            raw_samples = list(read_file_samples(file_path))
            for raw_sample in raw_samples:
                lang = raw_sample["language"]
                tokens, special_tokens = parser.parse_full(lang, raw_sample["code"])

                all_special_tokens.update(special_tokens)

                lengths[lang].append(len(tokens))

        end = time.time()
        logger.debug(f"all_special_tokens ({len(all_special_tokens)}) {all_special_tokens}")

        if not os.path.exists(pickle_path):
            os.makedirs(pickle_path)

        json_file = Path(pickle_path) / f"{name}_special_tokens.json"
        with open(json_file, "w") as f:
            json.dump(list(all_special_tokens), f)

        import statistics

        for lang, lgs in lengths.items():
            if len(lgs) > 0:
                max_lg = max(lgs)
                min_lg = min(lgs)
                mean_lg = statistics.mean(lgs)
                std_lg = statistics.stdev(lgs)
                logger.debug(f"{lang} [ min:{min_lg}, max:{max_lg}, mean:{mean_lg}, stddev:{std_lg} ]")

        time_p = end - start
        logger.info(f"Building AST took: {time_p} sec")
