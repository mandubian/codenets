# Some

from tensorboardX import SummaryWriter
from pathlib import Path
import datetime
from loguru import logger
from typing import Dict

from tensorboard.backend.event_processing import event_accumulator


def tensorboard_event_accumulator(
    file: str,
    loaded_scalars: int = 0,  # load all scalars by default
    loaded_images: int = 4,  # load 4 images by default
    loaded_compressed_histograms: int = 500,  # load one histogram by default
    loaded_histograms: int = 1,  # load one histogram by default
    loaded_audio: int = 4,  # loads 4 audio by default
):
    """Read a Tensorboard event_accumulator from a file"""
    ea = event_accumulator.EventAccumulator(
        file,
        size_guidance={  # see below regarding this argument
            event_accumulator.COMPRESSED_HISTOGRAMS: loaded_compressed_histograms,
            event_accumulator.IMAGES: loaded_images,
            event_accumulator.AUDIO: loaded_audio,
            event_accumulator.SCALARS: loaded_scalars,
            event_accumulator.HISTOGRAMS: loaded_histograms,
        },
    )
    ea.Reload()
    return ea


class Tensorboard:
    """
    Tensorboard manager

    This manager is associated to a:

    - experiment
    - a unique ID for the current run (one experiment can be run many times)
    - groups of metrics (like "train" or "val")
    - sub-groups of metrics (like train/bash or val/epoch)
    """

    def __init__(self, experiment_id, output_dir="./runs", unique_id=None, flush_secs=10):
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir)
        if unique_id is None:
            unique_id = datetime.datetime.now().isoformat(timespec="seconds")
        self.path = self.output_dir / f"{experiment_id}_{unique_id}"
        logger.debug(f"Writing TensorBoard events locally to {self.path}")
        self.writers: Dict[str, SummaryWriter] = {}
        self.flush_secs = flush_secs

    def _get_writer(self, group: str = "") -> SummaryWriter:
        if group not in self.writers:
            logger.debug(f"Adding group {group} to writers ({self.writers.keys()})")
            self.writers[group] = SummaryWriter(f"{str(self.path)}_{group}", flush_secs=self.flush_secs)
        return self.writers[group]

    def add_scalars(self, metrics: dict, global_step: int, group=None, sub_group="") -> None:
        for key, val in metrics.items():
            cur_name = "/".join([sub_group, key])
            self._get_writer(group).add_scalar(cur_name, val, global_step)
