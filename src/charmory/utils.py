"""Armory Utilities for Public Use"""

from copy import deepcopy
from typing import Sequence

import PIL
from art.defences.postprocessor import Postprocessor
from art.defences.preprocessor import Preprocessor
from art.estimators import BaseEstimator
import numpy as np


def apply_art_postprocessor_defense(estimator: BaseEstimator, defense: Postprocessor):
    """
    Applies the given postprocessor defense to the model, handling the presence
    or absence of existing postprocessors

    Example::

        from art.defences.postprocessor import GaussianNoise
        from art.estimators.classification import PyTorchClassifier
        from charmory.utils import apply_art_postprocessor_defense

        classifier = PyTorchClassifier(...)
        defense = JpegCompression(...)
        apply_art_postprocessor_defense(classifier, defense)

    Args:
        estimator: ART estimator to which to apply the postprocessor defense
        defense: ART postprocessor defense to be applied to the model
    """
    defenses = estimator.get_params().get("postprocessing_defences")
    if defenses:
        defenses.append(defense)
    else:
        defenses = [defense]
    estimator.set_params(postprocessing_defences=defenses)


def apply_art_preprocessor_defense(estimator: BaseEstimator, defense: Preprocessor):
    """
    Applies the given preprocessor defense to the model, handling the presence
    or absence of existing preprocessors

    Example::

        from art.defences.preprocessor import JpegCompression
        from art.estimators.classification import PyTorchClassifier
        from charmory.utils import apply_art_preprocessor_defense

        classifier = PyTorchClassifier(...)
        defense = JpegCompression(...)
        apply_art_preprocessor_defense(classifier, defense)

    Args:
        estimator: ART estimator to which to apply the preprocessor defense
        defense: ART preprocessor defense to be applied to the model
    """
    defenses = estimator.get_params().get("preprocessing_defences")
    if defenses:
        defenses.append(defense)
    else:
        defenses = [defense]
    estimator.set_params(preprocessing_defences=defenses)


def create_jatic_dataset_transform(preprocessor, image_key="image"):
    """
    Create a transform function that can be applied to JATIC-wrapped datasets
    using the preprocessor from a JATIC-wrapped model.

    Example::

        from charmory.utils import create_jatic_dataset_transform
        from jatic_toolbox import load_dataset, load_model

        model = load_model(
            provider="huggingface",
            model_name="microsoft/resnet-18",
            task="image-classification",
        )
        transform = create_jatic_dataset_transform(model.preprocessor)

        dataset = load_dataset(
            provider="huggingface",
            dataset_name="cifar10",
            task="image-classification",
            split="test",
        )
        dataset.set_transform(transform)

    Args:
        preprocessor: JATIC-wrapped model preprocessor. A function that accepts
          the image data and returns a dictionary with an "image" key containing
          the processed image data (i.e., `{"image": [...] }`)
        image_key: Key for the image data in the input dataset sample dictionary

    Returns:
        A function that will transform a sample from a JATIC-wrapped image
        classification dataset for a particular model
    """

    def transform(sample):
        transformed = deepcopy(sample)
        if isinstance(sample[image_key], Sequence):
            transformed[image_key] = [
                img.numpy() for img in preprocessor(sample[image_key])["image"]
            ]
        else:
            transformed[image_key] = preprocessor([sample[image_key]])["image"][
                0
            ].numpy()
        return transformed

    return transform


def is_defended(estimator: BaseEstimator) -> bool:
    """
    Checks if the given estimator has any preprocessor or postprocessor defenses
    applied to it.

    Example::

        from art.estimators.classification import PyTorchClassifier
        from charmory.utils import is_defended

        classifier = PyTorchClassifier(...)
        if is_defended(classifier):
            pass

    Args:
        estimator: ART estimator to be checked for defenses

    Returns:
        True if ART estimator has defenses, else False
    """
    preprocessor_defenses = estimator.get_params().get("preprocessing_defences")
    if preprocessor_defenses:
        return True
    postprocessor_defenses = estimator.get_params().get("postprocessing_defences")
    if postprocessor_defenses:
        return True
    return False


class PILtoNumpy(object):
    """
    Custom torchvision transform that converts PIL images to numpy arrays

    Example::

        training_data = torchvision.datasets.Food101(
        root="some/root/location",
        split = "train",
        download=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.Resize(512,512), PILtoNumpy()])

    Args:
        the __call__ method takes a sample of type PIL.Image.Image
    Returns:
        the sample PIL Image converted to a numpy array.
    """

    def __call__(self, sample):
        assert isinstance(sample, PIL.Image.Image)
        np_image = np.array(sample)
        return np_image


class PILtoNumpy_HuggingFace(object):
    """
    Custom torchvision transform a HuggingFace Dataset dictionary which
    contains a PIL images and converts the PIL Image to a numpy array

    Example::

        transform=PILtoNumpy_HuggingFace()


        train_dataset = load_jatic_dataset(
            provider="huggingface",
            dataset_name="keremberke/pokemon-classification",
            task="image-classification",
            name='full',
            split="train"
        )

        train_dataset.set_transform(transform)
    Args:
        the __call__ method takes a sample of type dict "{"image": [...],"label": [...] }".
        It converts the dict location "image" which is PIL Image to a numpy array
    Returns:
        the sample dict with converted PIL Image to numpy array.
    """

    def __call__(self, sample):
        sample["image"] = [np.asarray(img) for img in sample["image"]]
        return sample


class PILtoNumpy_HuggingFace_Variable_Length(object):
    """
    Custom torchvision transform a HuggingFace Dataset dictionary which
    contains a PIL images and converts the PIL Image to a numpy array
    with variable length images

    Example::

        transform=PILtoNumpy_HuggingFace()


        train_dataset = load_jatic_dataset(
            provider="huggingface",
            dataset_name="keremberke/pokemon-classification",
            task="image-classification",
            name='full',
            split="train"
        )

        train_dataset.set_transform(transform)
    Args:
        the __call__ method takes a sample of type dict "{"image": [...],"label": [...] }".
        It converts the dict location "image" which is PIL Image to a numpy array with a set
        new image size of 500 by 500 pixels
    Returns:
        the sample dict with converted PIL Image to numpy array.
    """

    def __init__(self, size=(500, 500)):
        self.size = size

    def __call__(self, sample):
        sample["image"] = [img.resize(self.size) for img in sample["image"]]
        sample["image"] = [np.asarray(img) for img in sample["image"]]
        if len(sample["image"][0].shape) != 3:
            # Convert from black/white to RGB
            sample["image"] = [np.dstack((img, img, img)) for img in sample["image"]]
        sample["image"] = [img.transpose(2, 0, 1) for img in sample["image"]]

        return sample
