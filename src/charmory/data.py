"""Armory Dataset Classes"""

# This could get merged with armory.data.datasets

from typing import TYPE_CHECKING

import numpy as np
from torch.utils.data.dataloader import DataLoader

if TYPE_CHECKING:
    import jatic_toolbox.protocols

from charmory.track import track_init_params


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
class JaticVisionDataLoader(DataLoader):
    """
    Data loader for a JATIC image classification dataset.
    """

    def __init__(
        self,
        dataset: "jatic_toolbox.protocols.VisionDataset",
        batch_size: int = 1,
        shuffle: bool = False,
        image_key: str = "image",
        label_key: str = "label",
        **kwargs,
    ):
        kwargs.pop("collate_fn", None)
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_collate_image_classification(image_key, label_key),
            **kwargs,
        )


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
