import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Type, TypeVar, Union, MutableMapping, List
from loguru import logger
import torch
from torch import nn

from pyhocon import ConfigFactory, ConfigTree, HOCONConverter
from codenets.utils import full_classname, instance_full_classname, runtime_import

# The Type of a class that can be loaded using Recordable.load classmethod
Recordable_T = TypeVar("Recordable_T", bound="Recordable")


class Recordable:
    """A recordable is something that be saved or loaded from a directory"""

    def save(self, output_dir: Union[Path, str]) -> bool:
        pass

    @classmethod
    def load(cls: Type[Recordable_T], restore_dir: Union[Path, str]) -> Recordable_T:
        pass


class NoneRecordable(Recordable):
    def __init__(self):
        super(NoneRecordable, self).__init__()

    def save(self, output_dir: Union[Path, str]) -> bool:
        return True

    @classmethod
    def load(cls, restore_dir: Union[Path, str]) -> "NoneRecordable":
        return NoneRecordable()


Module_T = TypeVar("Module_T", bound="nn.Module")


class RecordableTorchModule(nn.Module, Recordable):
    """A classic Pytorch module that is recordable"""

    def save(self, output_dir: Union[Path, str]) -> bool:
        full_dir = Path(output_dir) / instance_full_classname(self)
        logger.debug(f"Saving {instance_full_classname(self)} instance to {full_dir}")
        os.makedirs(full_dir, exist_ok=True)
        torch.save(self.state_dict(), full_dir / "state_dict.pth")
        return True

    @classmethod
    def load(cls: Type[Module_T], restore_dir: Union[Path, str]) -> Module_T:
        full_dir = Path(restore_dir) / full_classname(cls)
        logger.debug(f"Loading {full_classname(cls)} from {full_dir}")
        state_dict = torch.load(full_dir / "state_dict.pth")
        module = cls()
        module.load_state_dict(state_dict)
        return module


def runtime_load_recordable(dir: Union[Path, str]) -> Recordable:
    """
    Runtime factory using the first subdirectory name as the Recordableclass name to instantiate
    
    Args:
        dir (Union[Path, str]): directory in which to search for the class name
    
    Returns:
        RecordableTorchModule: the Recordable instantiated from the subdirectory name
    """
    cls_name = os.listdir(dir)[0]
    # TODO Check cls_name is a Recordable subclass
    klass = runtime_import(cls_name)
    recordable = klass.load(Path(dir))
    return recordable


def runtime_load_recordable_module(dir: Union[Path, str]) -> RecordableTorchModule:
    """
    Runtime factory using the first subdirectory name as the RecordableTorchModule class name to instantiate
    
    Args:
        dir (Union[Path, str]): directory in which to search for the class name
    
    Returns:
        RecordableTorchModule: the Recordable instantiated from the subdirectory name
    """
    cls_name = os.listdir(dir)[0]
    # TODO Check cls_name is a Recordable subclass
    klass = runtime_import(cls_name)
    recordable = klass.load(Path(dir))
    return recordable


class HoconConfigRecordable(Recordable):
    def __init__(self, config: ConfigTree):
        super(HoconConfigRecordable, self).__init__()
        self.config = config

    def save(self, output_dir: Union[Path, str]) -> bool:
        d = Path(output_dir) / instance_full_classname(self)
        if not os.path.exists(d):
            os.makedirs(d)
        logger.info(f"Saving Config to {d / 'config.conf'}")
        with open(d / "config.conf", "w") as f:
            f.write(HOCONConverter.to_hocon(self.config))
        return True

    @classmethod
    def load(cls, restore_dir: Union[Path, str]) -> "HoconConfigRecordable":
        conf_file = Path(restore_dir) / full_classname(cls) / "config.conf"
        logger.info(f"Loading Config File from {conf_file}")
        conf: ConfigTree = ConfigFactory.parse_file(conf_file)
        return HoconConfigRecordable(conf)


class ConfigFileRecordable(Recordable):
    def __init__(self, conf_file: Union[str, Path]):
        super(ConfigFileRecordable, self).__init__()
        self.conf_file = conf_file

    def get_config(self):
        return ConfigFactory.parse_file(self.conf_file)

    def save(self, output_dir: Union[Path, str]) -> bool:
        fp = Path(self.conf_file)
        d = Path(output_dir) / instance_full_classname(self)
        if not os.path.exists(d):
            os.makedirs(d)
        logger.info(f"Saving Config File {self.conf_file} to {d}")

        shutil.copyfile(fp, d / fp.name)
        return True

    @classmethod
    def load(cls, restore_dir: Union[Path, str]) -> "ConfigFileRecordable":
        d = Path(restore_dir) / full_classname(cls)

        _, _, files = list(os.walk(d))[0]

        logger.info(f"Loading Config File {files[0]} from {d}")
        return ConfigFileRecordable(d / files[0])


class DictRecordable(Recordable, Dict):
    """A recordable for a basic Dict"""

    def __init__(self, state: Dict[str, Any]):
        super(DictRecordable, self).__init__(state)
        # self.state = state

    def save(self, output_dir: Union[Path, str]) -> bool:
        d = Path(output_dir) / instance_full_classname(self)
        if not os.path.exists(d):
            os.makedirs(d)
        logger.info(f"Saving State dict to {d}")

        # js = json.dumps(self.state)
        js = json.dumps(self)
        f = open(d / "state_dict.json", "w")
        f.write(js)
        f.close()
        # pickle.dump(self.state, open(d / "state_dict.txt", "w"))
        return True

    @classmethod
    def load(cls, restore_dir: Union[Path, str]) -> "DictRecordable":
        d = Path(restore_dir) / full_classname(cls)
        logger.info(f"Loading State dict from {d}")

        # state = pickle.load(open(d / "state_dict.txt", "r"))
        f = open(d / "state_dict.json", "r")
        state = json.loads(f.read())
        f.close()

        return DictRecordable(state)


def save_recordable_mapping(output_dir: Union[Path, str], records: Mapping[str, Recordable]) -> bool:
    d = Path(output_dir)
    for name, record in records.items():
        record.save(d / name)
    return True


def runtime_load_recordable_mapping(
    restore_dir: Union[Path, str], accepted_keys: List[str] = []
) -> Mapping[str, Recordable]:
    d = Path(restore_dir)
    records: MutableMapping[str, Recordable] = {}
    for subdir in sorted(os.listdir(d)):
        if len(accepted_keys) > 0 and subdir not in accepted_keys:
            logger.debug(f"skipping recordables from {subdir}")
            continue
        logger.info(f"Loading {d / subdir}")
        records[subdir] = runtime_load_recordable(d / subdir)
    return records


RecordableMapping_T = TypeVar("RecordableMapping_T", bound="RecordableMapping")


class RecordableMapping(Recordable, Dict):
    def __init__(self, records: Mapping[str, Recordable]):
        super(RecordableMapping, self).__init__(records)
        # self.records = records

    def save(self, output_dir: Union[Path, str]) -> bool:
        d = Path(output_dir) / instance_full_classname(self)
        for name, record in self.items():
            logger.debug(f"RecordableMapping - Saving {name}")
            record.save(d / name)
        return True

    @classmethod
    def load(cls: Type[RecordableMapping_T], restore_dir: Union[Path, str]) -> RecordableMapping_T:
        d = Path(restore_dir) / full_classname(cls)
        records: Dict[str, Recordable] = {}  # OrderedDict()
        for subdir in sorted(os.listdir(d)):
            logger.debug(f"RecordableMapping - Loading {subdir}")
            records[subdir] = runtime_load_recordable(d / subdir)
        # return cls.from_dict_recordable(records)
        return cls(records)


RecordableTorchModuleMapping_T = TypeVar("RecordableTorchModuleMapping_T", bound="RecordableTorchModuleMapping")


class RecordableTorchModuleMapping(nn.ModuleDict, Recordable):
    # Can't inherit from RecordableMapping because it inherits Dict
    # and ModuleDict vs Dict colldies
    def __init__(self, records: Mapping[str, RecordableTorchModule]):
        # Forcing calls of super __init__ as visible python can go that far with super-typing
        nn.ModuleDict.__init__(self, records)
        self.records = records
        # RecordableMapping.__init__(self, records)

    def save(self, output_dir: Union[Path, str]) -> bool:
        d = Path(output_dir) / instance_full_classname(self)
        for name, record in self.records.items():
            record.save(d / name)
        return True

    @classmethod
    def load(
        cls: Type[RecordableTorchModuleMapping_T], restore_dir: Union[Path, str]
    ) -> RecordableTorchModuleMapping_T:
        d = Path(restore_dir) / full_classname(cls)
        records: Dict[str, RecordableTorchModule] = {}  # OrderedDict()
        for subdir in sorted(os.listdir(d)):
            logger.debug(f"Loading {subdir}")
            records[subdir] = runtime_load_recordable_module(d / subdir)
        return cls(records)
