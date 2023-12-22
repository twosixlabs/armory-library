"""Armory Dataset Classes"""

from typing import Any, Callable, List, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

import charmory.data as data
from charmory.track import track_init_params

DatasetOutputAdapter = Callable[..., Mapping[str, Any]]
"""
An adapter for dataset samples. The output must be a dictionary of column names
to values.
"""


class ArmoryDataset(Dataset):
    """
    Wrapper around a PyTorch dataset to apply an adapter to all samples obtained
    from the dataset.

    Example::

        from charmory.dataset import ArmoryDataset

        def rename_fields(sample):
            # Rename the 'data' field in the sample to 'image'
            sample["image"] = sample.pop("data")
            return sample

        # assuming `dataset` has been defined elsewhere
        renamed_dataset = ArmoryDataset(dataset, rename_fields)
    """

    def __init__(self, dataset, adapter: DatasetOutputAdapter):
        """
        Initializes the dataset.

        Args:
            dataset: Source dataset to be wrapped. It must be subscriptable and
                support the `len` operator.
            adapter: Dataset sample adapter
        """
        self._dataset = dataset
        self._adapter = adapter

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        return self._adapter(self._dataset[index])


class TupleDataset(ArmoryDataset):
    """
    Dataset wrapper with a pre-applied adapter to adapt tuples to map-like
    samples.

    Example::

        from charmory.dataset import TupleDataset

        # assuming `dataset` has been defined elsewhere
        print(dataset[0])
        # output: [[0, 0, 0], [0, 0, 0]], [5]

        tuple_ds = TupleDataset(dataset, x_key="image", y_key="label")
        print(tuple_ds[0])
        # output: {'image': [[0, 0, 0], [0, 0, 0]], 'label': [5]}
    """

    def __init__(
        self,
        dataset,
        x_key: str,
        y_key: str,
    ):
        """
        Initializes the dataset.

        Args:
            dataset: Source dataset where samples are a two-entry tuple of data,
                or x, and target, or y.
            x_key: Key name to use for x data in the adapted sample dictionary
            y_key: Key name to use for y data in the adapted sample dictionary
        """
        super().__init__(dataset, self._adapt)
        self._x_key = x_key
        self._y_key = y_key

    def _adapt(self, sample):
        x, y = sample
        return {self._x_key: x, self._y_key: y}


def _collate_by_type(values: List):
    if len(values) == 0:
        return []
    if isinstance(values[0], np.ndarray):
        return np.asarray(values)
    if isinstance(values[0], torch.Tensor):
        return torch.stack(values)
    return values


def _pop_and_cast(values, key):
    value = values.pop(key)
    if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
        return value
    if isinstance(value, list):
        return np.asarray(value)
    raise ValueError(f"Dataset {key} is unsupported type: {type(value)}")


@track_init_params
class ImageClassificationDataLoader(DataLoader):
    def __init__(
        self,
        *args,
        dim: data.ImageDimensions,
        image_key: str,
        label_key: str,
        scale: data.Scale,
        **kwargs,
    ):
        kwargs.pop("collate_fn", None)
        super().__init__(*args, collate_fn=self._collate, **kwargs)
        self.dim = dim
        self.image_key = image_key
        self.label_key = label_key
        self.scale = scale

    def _collate(self, samples: Sequence[Mapping[str, Any]]) -> data.Batch:
        collated = {
            key: _collate_by_type([s[key] for s in samples])
            for key in samples[0].keys()
        }
        images = data.Images(
            images=_pop_and_cast(collated, self.image_key),
            dim=self.dim,
            scale=self.scale,
        )
        labels = data.NDimArray(_pop_and_cast(collated, self.label_key))
        return data.ImageClassificationBatch(
            inputs=images,
            targets=labels,
            metadata=data.Metadata(data=collated, perturbations=dict()),
        )
