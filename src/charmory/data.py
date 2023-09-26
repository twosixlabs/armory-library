"""Armory Dataset Classes"""

# This could get merged with armory.data.datasets

from typing import TYPE_CHECKING, Any, Callable, Tuple

import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

if TYPE_CHECKING:
    import jatic_toolbox.protocols

from charmory.track import track_init_params

DatasetOutputAdapter = Callable[..., Tuple[Any, Any]]
"""
An adapter for dataset samples. The output must be a tuple of sample data and
label data.
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


class MapSampleDataset(ArmoryDataset):
    """Dataset wrapper with a pre-applied adapter to adapt map-like samples to tuples"""

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
        x = sample[self._x_key]
        y = sample[self._y_key]
        return x, y


class JaticImageClassificationDataset(MapSampleDataset):
    """Dataset wrapper with a pre-applied adapter for JATIC image classification datasets"""

    def __init__(
        self,
        dataset: "jatic_toolbox.protocols.VisionDataset",
    ):
        super().__init__(dataset, x_key="image", y_key="label")


class JaticObjectDetectionDataset(MapSampleDataset):
    """Dataset wrapper with a pre-applied adapter for JATIC image classification datasets"""

    def __init__(
        self,
        dataset: "jatic_toolbox.protocols.ObjectDetectionDataset",
    ):
        super().__init__(dataset, x_key="image", y_key="objects")


class ArmoryDataLoader(DataLoader):
    """
    Customization of the PyTorch DataLoader to produce numpy arrays instead of
    Tensors, as required by ART
    """

    def __init__(self, *args, **kwargs):
        kwargs.pop("collate_fn", None)
        super().__init__(*args, collate_fn=self._collate, **kwargs)

    @staticmethod
    def _collate(batch):
        x, y = zip(*batch)
        return np.asarray(x), np.asarray(y)


def _collate_image_classification(image_key, label_key):
    """Create a collate function that works with image classification samples"""

    def collate(batch):
        x = np.asarray([sample[image_key] for sample in batch])
        y = np.asarray([sample[label_key] for sample in batch])
        return x, y

    return collate


def _collate_object_detection(image_key, objects_key):
    """Create a collate function that works with object detection samples"""

    def collate(batch):
        x = np.asarray([sample[image_key] for sample in batch])
        y = [sample[objects_key] for sample in batch]
        return x, y

    return collate


@track_init_params()
class JaticObjectDetectionDataLoader(DataLoader):
    """
    Data loader for a JATIC object detection dataset.
    """

    def __init__(
        self,
        dataset: "jatic_toolbox.protocols.ObjectDetectionDataset",
        batch_size: int = 1,
        shuffle: bool = False,
        image_key: str = "image",
        objects_key: str = "objects",
        **kwargs,
    ):
        kwargs.pop("collate_fn", None)
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_collate_object_detection(image_key, objects_key),
            **kwargs,
        )
