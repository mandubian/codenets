import numpy as np
from typing import Tuple
from scipy.spatial.distance import cdist


def compute_ranks(
    src_representations: np.ndarray, tgt_representations: np.ndarray, distance_metric: str
) -> Tuple[np.array, np.array]:
    distances = cdist(src_representations, tgt_representations, metric=distance_metric)
    # By construction the diagonal contains the correct elements
    correct_elements = np.expand_dims(np.diag(distances), axis=-1)
    return np.sum(distances <= correct_elements, axis=-1), distances
