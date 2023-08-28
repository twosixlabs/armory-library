"""Armory Utilities for Public Use"""

from copy import deepcopy
from typing import Sequence

import PIL
from art.defences.postprocessor import Postprocessor
from art.defences.preprocessor import Preprocessor
from art.estimators import BaseEstimator
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


def adapt_jatic_object_detection_model_for_art(model):
    orig_forward = model.forward

    def patched_forward(data, *args):
        result = orig_forward(data)
        predictions = []
        for item in zip(result.boxes, result.labels, result.scores):
            predictions.append(dict(boxes=item[0], labels=item[1], scores=item[2]))
        return predictions

    model.forward = patched_forward


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
