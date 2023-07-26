"""Armory Utilities for Public Use"""

from copy import deepcopy
from typing import Sequence

import PIL
import numpy as np


def adapt_jatic_image_classification_model_for_art(model):
    """
    Adapts the given JATIC-wrapped image classification model so that it is
    compatible with the adversarial robustness toolkit (ART).

    JATIC-wrapped models return a structured object as their output. However,
    ART expects all models to return the predicted values as a tensor. We make
    the model compatible by monkey-patching the `forward` method to return the
    predicted values from the structured return object.

    Example::

        from art.estimators.classification import PyTorchClassifier
        from charmory.utils import adapt_jatic_image_classification_model_for_art
        from jatic_toolbox import load_model

        model = load_model(
            provider="huggingface",
            model_name="microsoft/resnet-18",
            task="image-classification",
        )
        adapt_jatic_image_classification_model_for_art(model)
        classifier = PyTorchClassifier(model, ...)

    Args:
        model: JATIC-wrapped image classification model
    """

    orig_forward = model.forward

    def patched_forward(data):
        result = orig_forward(data)
        # temporary hack because the HuggingFace and TorchVision
        # JATIC-toolbox wrappers return different types
        # TODO remove the try/except when JATIC-toolbox updates the
        # return type for TorchVision models
        try:
            return result.probs
        except AttributeError:
            return result.logits

    model.forward = patched_forward


def create_jatic_image_classification_dataset_transform(
    preprocessor, image_key="image"
):
    """
    Create a transform function that can be applied to JATIC-wrapped image
    classification datasets using the preprocessor from a JATIC-wrapped model.

    Example::

        from charmory.utils import create_jatic_image_classification_dataset_transform
        from jatic_toolbox import load_dataset, load_model

        model = load_model(
            provider="huggingface",
            model_name="microsoft/resnet-18",
            task="image-classification",
        )
        transform = create_jatic_image_classification_dataset_transform(model.preprocessor)

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
        # temporary hack because the JATIC-toolbox HuggingFace model
        # wrapper doesn't work with non-sequence data
        # TODO remove the else-block when JATIC-toolbox updates the HuggingFace
        # model wrapper to support non-sequence data in the preprocessor
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
