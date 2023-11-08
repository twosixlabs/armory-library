"""Experimental, generalized data transform utilities."""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import albumentations as A
import numpy as np
import torch
from torchvision.ops import box_convert

###
# Types
###


@dataclass
class _BboxFormatNames:
    """Bounding box format names for torchvision and albumentations"""

    torchvision: str
    albumentations: str


class BboxFormat(Enum):
    """
    Supported bounding box formats. Enum members are included for both the
    torchvision and the albumentations names for ease of use.

    Attributes:
        XYWH: Bounding box format is (xmin, ymin, width, height)
        COCO: Bounding box format is (xmin, ymin, width, height)
        XYXY: Bounding box format is (xmin, ymin, xmax, ymax)
        PASCAL_VOC: Bounding box format is (xmin, ymin, xmax, ymax)
        CXCYWH: Bounding box format is (center x, center y, width, height)
        YOLO: Bounding box format is (center x, center y, width, height)
    """

    # (xmin, ymin, width, height)
    XYWH = _BboxFormatNames("xywh", "coco")
    COCO = _BboxFormatNames("xywh", "coco")
    # (xmin, ymin, xmax, ymax)
    XYXY = _BboxFormatNames("xyxy", "pascal_voc")
    PASCAL_VOC = _BboxFormatNames("xyxy", "pascal_voc")
    # (center x, center y, width, height)
    CXCYWH = _BboxFormatNames("cxcywh", "yolo")
    YOLO = _BboxFormatNames("cxcywh", "yolo")


Sample = Dict[str, Any]
"""A dataset sample or batch."""
Transform = Callable[[Sample], Sample]
"""A callable to transform a sample or batch."""


###
# Functions
###


def create_image_transform(
    max_size: Optional[int] = None,
    float_max_value: Optional[Union[int, float]] = False,
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
    **kwargs,
) -> A.Compose:
    """
    Creates an image transform capable of performing the following operations:

    - Resizing & padding (if a `max_size` is provided)
    - Rescaling values from 0-255 to 0-1 (if a `float_max_value` is provided)
    - Normalization into z-scores (if both `mean` and `std` are provided)

    Example::

        from charmory.experimental.transforms import create_image_transform
        import numpy as np

        transform = create_image_transform(max_size=256)
        result = transform(image=np.random.rand(400, 400, 3))
        image = result["image"]

    Args:
        max_size: Maximum width or height to which the image will be resized and
            padded. When omitted, no resizing will be performed.
        float_max_value: Maximum possible input value with which to divide pixel
            values to rescale into a range of 0.0 to 1.0. When omitted, no
            rescaling will be performed.
        mean: Tuple of mean values for each channel to be used for normalization
            into z-scores. When omitted, no normalization will be performed.
        std: Tuple of standard deviation values for each channel to be used for
            normalization into z-scores. When omitted, no normalization will be
            performed.
        **kwargs: All other keyword arguments will be forwarded to the
            `albumentations.Compose` class.

    Returns:
        Albumentations transform instance
    """
    transforms = []
    if max_size is not None:
        transforms.extend(
            [
                A.LongestMaxSize(max_size=max_size),
                A.PadIfNeeded(
                    min_height=max_size,
                    min_width=max_size,
                    border_mode=0,
                    value=(0, 0, 0),
                ),
            ]
        )
    if float_max_value:
        transforms.append(A.ToFloat(max_value=float_max_value))
    if mean is not None and std is not None:
        transforms.append(A.Normalize(mean=mean, std=std))
    return A.Compose(transforms, **kwargs)


def create_image_bbox_transform(
    format: BboxFormat,
    label_fields: Optional[List[str]] = None,
    min_area: float = 0,
    min_visibility: float = 0,
    min_width: float = 0,
    min_height: float = 0,
    **kwargs,
) -> A.Compose:
    """
    Creates an image and bounding box transform. See `create_image_transform`
    for all capabilities and arguments to configure the image transformation.

    Example::

        from charmory.experimental.transforms import (
            BboxFormat,
            create_image_bbox_transform,
        )
        import numpy as np

        transform = create_image_transform(max_size=256, format=BboxFormat.XYXY)
        result = transform(
            image=np.random.rand(400, 400, 3),
            bboxes=np.random.rand(3, 4) * 400,
        )
        image = result["image"]
        boxes = result["bboxes"]  # In XYXY format

    Args:
        format: Format of the input bounding boxes
        label_fields: List of fields that are joined with the boxes, e.g. labels
        min_area: Minimum area of a bounding box. All bounding boxes whose
            visible area in pixels is less than this value will be removed.
        min_visibility: Minimum fraction of area for a bounding box to remain.
        min_width: Minimum width of a bounding box. All bounding boxes whose
            width is less than this value will be removed.
        min_height: Minimum height of a bounding box. All bounding boxes whose
            height is less than this value will be removed.
        **kwargs: All other keyword arguments will be forwarded to the
            `create_image_transform` function.
    """
    return create_image_transform(
        bbox_params=A.BboxParams(
            format=format.value.albumentations,
            label_fields=label_fields,
            min_area=min_area,
            min_visibility=min_visibility,
            min_width=min_width,
            min_height=min_height,
        ),
        **kwargs,
    )


def convert_boxes(
    boxes: np.ndarray, from_format: BboxFormat, to_format: BboxFormat
) -> np.ndarray:
    """
    Convert bounding boxes from one format to another.

    Example::

        from charmory.experimental.transforms import BboxFormat, convert_boxes
        import numpy as np

        boxes = convert_boxes(np.random.rand(3, 4), BboxFormat.COCO, BboxFormat.XYXY)

    Args:
        boxes: A (N,4) numpy array of bounding boxes
        from_format: Input bounding box format
        to_format: Output bounding box format

    Returns:
        (N, 4) numpy array of bounding boxes in output format
    """
    if from_format.value.torchvision == to_format.value.torchvision:
        return boxes
    return box_convert(
        torch.tensor(boxes), from_format.value.torchvision, to_format.value.torchvision
    ).numpy()


def default_transpose(img: np.ndarray) -> np.ndarray:
    """Transposes the input image array from (H,W,C) to (C,H,W)"""
    return img.transpose(2, 0, 1)


def create_image_classification_transform(
    image_to_np: Callable[..., np.ndarray] = np.asarray,
    image_from_np: Callable[[np.ndarray], Any] = default_transpose,
    image_key: str = "image",
    preprocessor: Optional[Transform] = None,
    postprocessor: Optional[Transform] = None,
    **kwargs,
) -> Transform:
    """
    Creates a sample or batch transform capable of performing the following operations:

    - Image transformations
    - Arbitrary (user-supplied) pre and post transforms

    See `create_image_transform` for additional arguments.

    Example::

        from charmory.experimental.transforms import create_image_classification_transform
        import numpy as np

        transform = create_image_classification_transform(max_size=256)
        sample = transform(
            dict(
                image=[np.random.rand(400, 400, 3)],
                labels=[2],
            )
        )
        image = sample["image"][0]  # A CHW numpy array
        label = sample["labels"][0]

    Args:
        image_to_np: Callable to convert the input image to a numpy array
        image_from_np: Callable to convert the augmented image numpy array (from
            albumentations) to the output image type
        image_key: Key in the input batch dictionary for the images. Defaults to
            "image".
        preprocessor: Optional, arbitrary transform to apply to the sample prior
            to performing image transforms.
        postprocessor: Optional, arbitrary transform to apply to final output
            sample.
        **kwargs: All other keyword arguments will be forwarded to the
            `create_image_transform` function.

    Returns:
        Sample transform function
    """
    img_transform = create_image_transform(**kwargs)

    def transform_image(img):
        res = img_transform(image=image_to_np(img))
        return image_from_np(res["image"])

    def transform(sample: Sample) -> Sample:
        if preprocessor is not None:
            sample = preprocessor(sample)

        transformed = dict(**sample)
        transformed[image_key] = [transform_image(img) for img in sample[image_key]]

        if postprocessor is not None:
            transformed = postprocessor(transformed)

        return transformed

    return transform


def create_object_detection_transform(
    format: BboxFormat,
    image_to_np: Callable[..., np.ndarray] = np.asarray,
    image_from_np: Callable[[np.ndarray], Any] = default_transpose,
    image_key: str = "image",
    objects_key: str = "objects",
    bbox_key: str = "bbox",
    target_format: BboxFormat = BboxFormat.XYXY,
    rename_object_fields: Optional[Dict[str, str]] = None,
    preprocessor: Optional[Transform] = None,
    postprocessor: Optional[Transform] = None,
    **kwargs,
) -> Transform:
    """
    Creates a sample or batch transform capable of performing the following operations:

    - Image and bounding box transformations
    - Bounding box format conversion
    - Bounding box field renames
    - Arbitrary (user-supplied) pre and post transforms

    See `create_image_bbox_transform` for additional arguments.

    Example::

        from charmory.experimental.transforms import (
            BboxFormat,
            create_object_detection_transform,
        )
        import numpy as np

        transform = create_object_detection_transform(
            max_size=256,
            format=BboxFormat.XYWH,
            label_fields=["label"],
        )
        sample = transform(
            dict(
                image=[np.random.rand(400, 400, 3)],
                objects=[
                    dict(
                        bbox=np.random.rand(3, 4) * 400,
                        label=[1, 2, 0],
                    )
                ]
            )
        )
        image = sample["image"][0]  # A CHW numpy array
        objects = sample["objects"][0]
        boxes = objects["bbox"]  # XYXY boxes
        labels = objects["label"]

    Args:
        image_to_np: Callable to convert the input image to a numpy array
        image_from_np: Callable to convert the augmented image numpy array (from
            albumentations) to the output image type
        image_key: Key in the input batch dictionary for the images. Defaults to
            "image".
        objects_key: Key in the input batch dictionary for the objects. Defaults
            to "objects".
        bbox_key: Key in the input objects dictionary for the bounding boxes.
            Defaults to "bbox".
        target_format: Desired output bounding box format
        rename_object_fields: Optional, mapping of original object field names
            to desired field names (e.g., to rename "bbox" to "boxes")
        preprocessor: Optional, arbitrary transform to apply to the sample prior
            to performing image and bounding box transforms.
        postprocessor: Optional, arbitrary transform to apply to final output
            sample.
        **kwargs: All other keyword arguments will be forwarded to the
            `create_image_bbox_transform` function.

    Returns:
        Sample transform function
    """
    img_bbox_transform = create_image_bbox_transform(format=format, **kwargs)

    def transform(sample: Sample) -> Sample:
        if preprocessor is not None:
            sample = preprocessor(sample)

        transformed = dict(**sample)
        transformed[image_key] = []
        transformed[objects_key] = []

        for idx in range(len(sample[image_key])):
            args: Sample = dict(
                image=image_to_np(sample[image_key][idx]),
                bboxes=sample[objects_key][idx][bbox_key],
            )
            # Include any "label" fields (albumentations will drop entries for
            # any boxes that get cut)
            for label_field in kwargs.get("label_fields", []):
                args[label_field] = sample[objects_key][idx][label_field]

            # Perform transform on the image+boxes
            res = img_bbox_transform(**args)

            transformed[image_key].append(image_from_np(res["image"]))

            # Re-construct remaining, transformed objects
            obj = dict()
            # Convert boxes if necessary
            obj[bbox_key] = convert_boxes(
                res["bboxes"], from_format=format, to_format=target_format
            )
            # Add any "label" fields back
            for label_field in kwargs.get("label_fields", []):
                obj[label_field] = res[label_field]
            # Perform any field renames as needed
            if rename_object_fields is not None:
                for old_name, new_name in rename_object_fields.items():
                    obj[new_name] = obj[old_name]
                    del obj[old_name]
            transformed[objects_key].append(obj)

        if postprocessor is not None:
            transformed = postprocessor(transformed)

        return transformed

    return transform
