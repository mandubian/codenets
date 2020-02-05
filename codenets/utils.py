from typing import Iterable, List, Optional
import os
import re
from dpu_utils.codeutils import split_identifier_into_parts

# from dpu_utils.utils import Path
from pathlib import Path
import numpy as np
import glob

IDENTIFIER_TOKEN_REGEX = re.compile("[_a-zA-Z][_a-zA-Z0-9]*")


def runtime_import(class_name: str):
    """
    Runtime import from a string using "." to split module & class names
    
    Args:
        class_name (str): the class name to split according to "." and load dynamically modules & class
    
    Returns:
        Class: The imported class
    """
    components = class_name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def full_classname(cls):
    return cls.__module__ + "." + cls.__name__


def instance_full_classname(o):
    # o.__module__ + "." + o.__class__.__qualname__ is an example in
    # this context of H.L. Mencken's "neat, plausible, and wrong."
    # Python makes no guarantees as to whether the __module__ special
    # attribute is defined, so we take a more circumspect approach.
    # Alas, the module name is explicitly excluded from __qualname__
    # in Python 3.
    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__  # Avoid reporting __builtin__
    else:
        return module + "." + o.__class__.__name__


def _to_subtoken_stream(input_stream: Iterable[str], mark_subtoken_end: bool) -> Iterable[str]:
    for token in input_stream:
        if IDENTIFIER_TOKEN_REGEX.match(token):
            yield from split_identifier_into_parts(token)
            if mark_subtoken_end:
                yield "</id>"
        else:
            yield token


def expand_data_path(data_path: str) -> List[Path]:
    """
    Args:
        data_path: A path to either a file or a directory. If it's a file, we interpret it as a list of
            data directories.

    Returns:
        List of data directories (potentially just data_path)
    """
    data_rpath = Path(data_path)

    if data_rpath.is_dir():
        return [data_rpath]

    data_dirs: List[Path] = []
    with open(data_rpath) as f:
        for fl in map(Path, f.read().splitlines()):
            if fl.is_absolute():
                data_dirs.append(fl)
            else:
                data_dirs.append(data_rpath.parent / fl)

        # data_dirs.extend(map(Path))
    return data_dirs


def get_data_files_from_directory(data_dirs: List[Path], max_files_per_dir: Optional[int] = None) -> List[Path]:
    files: List[Path] = []
    for data_dir in data_dirs:
        dir_files = [Path(path) for path in glob.iglob(os.path.join(data_dir, "*.jsonl.gz"), recursive=True)]
        # dir_files = data_dir.get_filtered_files_in_dir("*.jsonl.gz")
        if max_files_per_dir:
            dir_files = sorted(dir_files)[: int(max_files_per_dir)]
        files += dir_files

    np.random.shuffle(files)  # This avoids having large_file_0, large_file_1, ... subsequences
    return files
