import os
from pathlib import Path
import shutil
from typing import Union, Type, TypeVar, Optional
from codenets.recordable import Recordable


def rotating_save_records(path: Union[Path, str], prefix: str, rec: Recordable, nb: int = 5) -> bool:
    root_path = Path(path) / prefix
    if not os.path.isdir(root_path):
        os.makedirs(root_path)

    paths = []
    first_empty_path = None
    saved = True
    for i in range(nb):
        path_i = root_path / f"{prefix}_{i}"
        if not os.path.exists(path_i) and first_empty_path is None:
            first_empty_path = path_i
            os.makedirs(first_empty_path)
        paths.append(path_i)

    if first_empty_path is not None:
        saved = saved and rec.save(first_empty_path)
    else:
        first = paths[0]

        shutil.rmtree(first)
        for pth in paths[1:]:
            os.rename(pth, first)
            first = pth
        saved = saved and rec.save(paths[-1])

    return saved


def save_records_direct(path: Union[Path, str], rec: Recordable) -> bool:
    if not os.path.isdir(path):
        os.makedirs(path)

    return rec.save(path)


def save_records_best(path: Union[Path, str], rec: Recordable) -> bool:
    prefix = os.path.basename(path)
    best_path = Path(path) / f"{prefix}_best"
    if not os.path.isdir(best_path):
        os.makedirs(best_path)

    return rec.save(best_path)


def save_records_last(output_dir: Union[Path, str], rec: Recordable) -> bool:
    return rotating_save_records(os.path.dirname(output_dir), os.path.basename(output_dir), rec)


Recordable_T = TypeVar("Recordable_T", bound="Recordable")


def rotating_recover_records(
    cls: Type[Recordable_T], path: Union[Path, str], prefix: str, nb: int = 5
) -> Optional[Recordable_T]:
    last_path = None
    for i in range(nb):
        path_i = Path(path) / prefix / f"{prefix}_{i}"
        if os.path.exists(path_i):
            last_path = path_i

    if last_path is not None:
        return cls.load(last_path)
    else:
        return None


def recover_records_best(
    cls: Type[Recordable_T], recover_dir: Union[Path, str], nb: int = 5, *args, **kwargs
) -> Optional[Recordable_T]:
    prefix = os.path.basename(recover_dir)
    best_path = Path(recover_dir) / f"{prefix}_best"
    if best_path.exists():
        return cls.load(best_path)
    else:
        return None


def recover_records_direct(
    cls: Type[Recordable_T], recover_dir: Union[Path, str], *args, **kwargs
) -> Optional[Recordable_T]:
    p = Path(recover_dir)
    if p.exists():
        return cls.load(p)
    else:
        return None


def recover_records_last(cls: Type[Recordable_T], recover_dir: Union[Path, str]) -> Optional[Recordable_T]:
    return rotating_recover_records(cls, os.path.dirname(recover_dir), os.path.basename(recover_dir))
