"""Armory Dataset Classes"""

# This could get merged with armory.data.datasets

import numpy as np
from torch.utils.data.dataloader import DataLoader

from armory.data.datasets import ArmoryDataGenerator


class _DataLoaderGenerator:
    """
    Iterable wrapper around a pytorch data loader to enable infinite iteration (required by ART)
    """

    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(self.loader)

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            # Reset when we reach the end of the iterator/epoch
            self.iterator = iter(self.loader)
            batch = next(self.iterator)
        return batch


def _collate_image_classification(image_key, label_key):
    """Create a collate function that works with image classification samples"""

    def collate(batch):
        x = np.asarray([sample[image_key] for sample in batch])
        y = np.asarray([sample[label_key] for sample in batch])
        return x, y

    return collate


class JaticVisionDatasetGenerator(ArmoryDataGenerator):
    """
    Data generator for a JATIC image classification dataset.
    """

    def __init__(
        self,
        dataset,
        epochs: int,
        batch_size=1,
        shuffle=False,
        image_key="image",
        label_key="label",
        preprocessing_fn=None,
        label_preprocessing_fn=None,
        context=None,
        size=None,
    ):
        super().__init__(
            generator=_DataLoaderGenerator(
                DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    collate_fn=_collate_image_classification(image_key, label_key),
                )
            ),
            size=size or len(dataset),
            batch_size=batch_size,
            epochs=epochs,
            preprocessing_fn=preprocessing_fn,
            label_preprocessing_fn=label_preprocessing_fn,
            context=context,
        )
