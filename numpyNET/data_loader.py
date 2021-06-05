""" To feed the data to our network in batches
"""
import numpy as np
from numpy import ndarray as Vector
from typing import Iterator, NamedTuple

# If "python version" >= 3.6 use:
class Batch(NamedTuple):
    features: Vector
    labels: Vector

# If 3.5 < "python version" < 3.6 use:
# Batch = NamedTuple('Batch', features=Vector, labels=Vector)

# If "python version" <= 3.5 use:
# Batch = NamedTuple("Batch", [("features", Vector), ("labels", Vector)])

class DataIterator:
    def __call__(self, features: Vector, labels: Vector) -> Iterator[Batch]:
        raise NotImplementedError


class BatchIterator(DataIterator):
    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, features: Vector, labels: Vector) -> Iterator[Batch]:
        starts = np.arange(0, len(features), self.batch_size)
        if self.shuffle:
            np.random.shuffle(starts)

        for start in starts:
            end = start + self.batch_size
            batch_features = features[start:end]
            batch_labels = labels[start:end]
            yield Batch(batch_features, batch_labels)
