"""Armory Dataset Classes"""

# This could get merged with armory.data.datasets

from armory.data.datasets import ArmoryDataGenerator
import numpy as np
from torch.utils.data.dataloader import DataLoader


def _image_transform(batch):
    if type(batch["image"]) == list:
        batch["image"] = [np.asarray(img) for img in batch["image"]]
    else:
        batch["image"] = np.asarray(batch["image"])

    return batch

class _InnerGenerator:
    """Iterable wrapper around a dataset that contains image and label features"""

    def __init__(self, dataset, shuffle, batch_size, image_key, label_key):
        dataset.set_transform(_image_transform)
        self.loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
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
