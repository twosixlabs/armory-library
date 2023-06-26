"""Armory Dataset Classes"""

# This could get merged with armory.data.datasets

import numpy as np

class JaticVisionDatasetGenerator:
    """
    Wrapper around a dataset that contains image and label features.
    """

    def __init__(self, dataset, batch_size = 1, image_key = "image", label_key = "label"):
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
