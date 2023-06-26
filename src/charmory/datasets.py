"""Armory Dataset Classes"""

# This could get merged with armory.data.datasets

from armory.data.datasets import ArmoryDataGenerator
import numpy as np

class _InnerGenerator:
    """Iterable wrapper around a dataset that contains image and label features"""

    def __init__(self, dataset, batch_size, image_key, label_key):
        self.dataset = dataset
        self.image_key = image_key
        self.label_key = label_key
        self.batch_size = batch_size
        self.current = 0

    def __next__(self):
        # Create batch
        stop = min(self.current + self.batch_size, len(self.dataset))
        x = []
        y = []
        for i in range(self.current, stop):
            sample = self.dataset[i]
            x.append(np.asarray(sample[self.image_key]))
            y.append(sample[self.label_key])

        # Reset the current index back to 0 when we reach the end so
        # that this iterator is indefinite
        self.current = stop
        if self.current == len(self.dataset):
            self.current = 0

        return np.asarray(x), np.asarray(y)


class JaticVisionDatasetGenerator(ArmoryDataGenerator):
    """
    Data generator for a JATIC image classification dataset.
    """

    def __init__(self,
        dataset,
        epochs: int,
        batch_size = 1,
        image_key = "image",
        label_key = "label",
        preprocessing_fn = None,
        label_preprocessing_fn = None,
        context = None,
    ):
        super().__init__(
            generator=_InnerGenerator(dataset, batch_size=batch_size, image_key=image_key, label_key=label_key),
            size=len(dataset),
            batch_size=batch_size,
            epochs=epochs,
            preprocessing_fn=preprocessing_fn,
            label_preprocessing_fn=label_preprocessing_fn,
            context=context,
        )
