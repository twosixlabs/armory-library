"""
Armory helper utilities to assist with use of HuggingFace datasets and models
with Armory
"""

from typing import TYPE_CHECKING, Optional, Any

from art.estimators.classification.pytorch import PyTorchClassifier
from transformers import AutoImageProcessor

from armory.data import DataType, ImageDimensions, Images, Scale
from armory.evaluation import ImageClassificationDataset
from armory.model.image_classification.image_classifier import ImageClassifier
from armory.track import track_init_params, track_params

if TYPE_CHECKING:
    import torch.nn

Sample = dict(str, Any)

def _create_scale_from_image_processor(processor: AutoImageProcessor) -> Scale:
    do_normalize = getattr(processor, "do_normalize", False)
    image_mean = getattr(processor, "image_mean", None)
    image_std = getattr(processor, "image_std", None)
    scale = Scale(
        dtype=DataType.FLOAT,
        max=1.0,
        mean=image_mean if do_normalize else None,
        std=image_std if do_normalize else None,
    )
    return scale


def _create_clip_values_from_scale(scale: Scale) -> (float, float):
    if not scale.is_normalized or not scale.mean or not scale.std:
        return (0, scale.max)
    else:
        min_val = 0.0 - max(scale.mean) / min(scale.std)
        max_val = scale.max - min(scale.mean) / min(scale.std)
        return (min_val, max_val)


def _transform(processor: AutoImageProcessor, sample: Sample) -> Sample:
    """Use the HF image processor and convert from BW To RGB"""
    sample["image"] = processor([img.convert("RGB") for img in sample["image"]])[
        "pixel_values"
    ]
    return sample


@track_params
def load_image_classification_dataset(
    name: str,
    split: str,
    processor: AutoImageProcessor,
    dim: ImageDimensions = ImageDimensions.CHW,
    image_key: str = "image",
    label_key: str = "label",
    **kwargs,
) -> ImageClassificationDataset:
    import functools

    import datasets

    from armory.dataset import ImageClassificationDataLoader
    from armory.evaluation import ImageClassificationDataset

    hf_dataset = datasets.load_dataset(name, split=split)
    assert isinstance(hf_dataset, datasets.Dataset)

    labels = hf_dataset.features[label_key].names

    hf_dataset.set_transform(functools.partial(_transform, processor))

    scale = _create_scale_from_image_processor(processor)

    dataloader = ImageClassificationDataLoader(
        hf_dataset,
        dim=dim,
        scale=scale,
        image_key=image_key,
        label_key=label_key,
        **kwargs,
    )

    evaluation_dataset = ImageClassificationDataset(
        name=name,
        dataloader=dataloader,
        labels=labels,
    )

    return evaluation_dataset


@track_params
def load_image_classification_model(
    name: str,
    loss: Optional["torch.nn.modules.loss._Loss"] = None,
    dim: ImageDimensions = ImageDimensions.CHW,
    num_channels: int = 3,
) -> tuple(ImageClassifier, PyTorchClassifier):
    from art.estimators.classification import PyTorchClassifier
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    from armory.model.image_classification import ImageClassifier

    hf_model = AutoModelForImageClassification.from_pretrained(name)

    hf_processor = AutoImageProcessor.from_pretrained(name)
    if hf_processor is None:
        raise RuntimeError(f"No image processor found for pretrained model, {name}")

    scale = _create_scale_from_image_processor(hf_processor)
    accessor = Images.as_torch(dim=dim, scale=scale)

    armory_model = ImageClassifier(
        name=name,
        model=hf_model,
        accessor=accessor,
    )

    if loss is None:
        import torch.nn

        loss = torch.nn.CrossEntropyLoss()

    channels_first = dim == ImageDimensions.CHW
    input_shape = (
        (num_channels, hf_processor.size["height"], hf_processor.size["width"])
        if channels_first
        else (hf_processor.size["height"], hf_processor.size["width"], num_channels)
    )
    clip_values = _create_clip_values_from_scale(scale)

    art_classifier = track_init_params(PyTorchClassifier)(
        armory_model,
        loss=loss,
        input_shape=input_shape,
        channels_first=channels_first,
        nb_classes=hf_model.num_labels,
        clip_values=clip_values,
    )

    return armory_model, art_classifier
