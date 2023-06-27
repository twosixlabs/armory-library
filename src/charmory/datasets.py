"""Armory Dataset Classes"""

# This could get merged with armory.data.datasets

from armory.data.datasets import ArmoryDataGenerator
import numpy as np
from torch.utils.data.dataloader import DataLoader


class _InnerGenerator:
    """Iterable wrapper around a dataset that contains image and label features"""

    def __init__(self, dataset, shuffle, batch_size, image_key, label_key):
        dataset.set_transform(self._transform_image)
        self.loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        self.iterator = iter(self.loader)
        self.image_key = image_key
        self.label_key = label_key

    def _transform_image(self, batch):
        """Transform PIL images to numpy arrays"""
        transformed_batch = dict(**batch)
        image = batch[self.image_key]
        if type(image) == list:
            transformed_batch[self.image_key] = [np.asarray(img) for img in image]
        else:
            transformed_batch[self.image_key] = np.asarray(image)
        return transformed_batch

    def __next__(self):
        try:
            sample = next(self.iterator)
        except StopIteration:
            # Reset when we reach the end of the iterator/epoch
            self.iterator = iter(self.loader)
            sample = next(self.iterator)

        return np.asarray(sample[self.image_key]), sample[self.label_key]


class JaticVisionDatasetGenerator(ArmoryDataGenerator):
    """
    Data generator for a JATIC image classification dataset.
    """

    def __init__(self,
        dataset,
        epochs: int,
        batch_size = 1,
        shuffle = False,
        image_key = "image",
        label_key = "label",
        preprocessing_fn = None,
        label_preprocessing_fn = None,
        context = None,
    ):
        super().__init__(
            generator=_InnerGenerator(dataset, shuffle=shuffle, batch_size=batch_size, image_key=image_key, label_key=label_key),
            size=len(dataset),
            batch_size=batch_size,
            epochs=epochs,
            preprocessing_fn=preprocessing_fn,
            label_preprocessing_fn=label_preprocessing_fn,
            context=context,
        )


class _DataloaderGenerator:
    """Iterable wrapper around a dataloader that contains image and label features"""

    def __init__(self, dataloader, image_key, label_key):
        self.loader = dataloader
        self.iterator = iter(self.loader)
        self.image_key = image_key
        self.label_key = label_key

    def __next__(self):
        try:
            sample = next(self.iterator)
        except StopIteration:
            # Reset when we reach the end of the iterator/epoch
            self.iterator = iter(self.loader)
            sample = next(self.iterator)

        x = np.asarray([np.asarray(img) for img in sample[self.image_key]])
        return x, sample[self.label_key]


class JaticVisionDataloaderGenerator(ArmoryDataGenerator):
    """
    Data generator for a JATIC image classification dataset wrapped in a data loader.
    """

    def __init__(self,
        dataloader,
        size,
        epochs,
        batch_size,
        image_key = "image",
        label_key = "label",
        preprocessing_fn = None,
        label_preprocessing_fn = None,
        context = None,
    ):
        super().__init__(
            generator=_DataloaderGenerator(dataloader, image_key, label_key),
            size=size,
            batch_size=batch_size,
            epochs=epochs,
            preprocessing_fn=preprocessing_fn,
            label_preprocessing_fn=label_preprocessing_fn,
            context=context,
        )