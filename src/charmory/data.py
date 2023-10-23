"""Armory Dataset Classes"""

from typing import Any, Callable, Mapping, Sequence

import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from charmory.track import track_init_params

DatasetOutputAdapter = Callable[..., Mapping[str, Any]]
"""
An adapter for dataset samples. The output must be a dictionary of column names
to values.
"""


class ArmoryDataset(Dataset):
    """Wrapper around a dataset to apply an adapter to all samples obtained from the dataset"""

    def __init__(self, dataset, adapter: DatasetOutputAdapter):
        self._dataset = dataset
        self._adapter = adapter

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        return self._adapter(self._dataset[index])


class TupleDataset(ArmoryDataset):
    """Dataset wrapper with a pre-applied adapter to adapt tuples to map-like samples"""

    def __init__(
        self,
        dataset,
        x_key: str,
        y_key: str,
    ):
        super().__init__(dataset, self._adapt)
        self._x_key = x_key
        self._y_key = y_key

    def _adapt(self, sample):
        x, y = sample
        return {self._x_key: x, self._y_key: y}


@track_init_params
class ArmoryDataLoader(DataLoader):
    """
    Customization of the PyTorch DataLoader to produce numpy arrays instead of
    Tensors, as required by ART
    """

    def __init__(self, *args, **kwargs):
        kwargs.pop("collate_fn", None)
        super().__init__(*args, collate_fn=self._collate, **kwargs)

    @staticmethod
    def _collate(batch: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        keys = list(batch[0].keys())
        collated = {}
        for key in keys:
            collated[key] = np.asarray([b[key] for b in batch])
        return collated
