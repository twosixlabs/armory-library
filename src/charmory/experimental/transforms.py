from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import albumentations as A
import numpy as np
import torch
from torchvision.ops import box_convert


@dataclass
class _BboxFormatNames:
    """Bounding box format names for torchvision and albumentations"""

    torchvision: str
    albumentations: str


class BboxFormat(Enum):
    """
    Supported bounding box formats. Enum members are included for both the
    torchvision and the albumentations names for ease of use.
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


def create_image_transform(
    max_size: Optional[int] = None,
    float_max_value: Optional[Union[int, float]] = False,
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
    **kwargs,
):
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
    label_fields: Optional[List[str]],
    min_area: float = 0,
    min_visibility: float = 0,
    min_width: float = 0,
    min_height: float = 0,
    **kwargs,
):
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
    if from_format.value.torchvision == to_format.value.torchvision:
        return boxes
    return box_convert(
        torch.tensor(boxes), from_format.value.torchvision, to_format.value.torchvision
    ).numpy()


def default_transpose(img: np.ndarray):
    # Transpose image from HWC to CHW
    return img.transpose(2, 0, 1)


Sample = Dict[str, Any]
SampleProcessor = Callable[[Sample], Sample]


def create_object_detection_transform(
    format: BboxFormat,
    img_to_np: Callable[..., np.ndarray] = np.asarray,
    img_from_np: Callable[[np.ndarray], Any] = default_transpose,
    image_key: str = "image",
    objects_key: str = "objects",
    bbox_key: str = "bbox",
    target_format: BboxFormat = BboxFormat.XYXY,
    rename_object_fields: Optional[Dict[str, str]] = None,
    preprocessor: Optional[SampleProcessor] = None,
    postprocessor: Optional[SampleProcessor] = None,
    **kwargs,
):
    img_bbox_transform = create_image_bbox_transform(format=format, **kwargs)

    def transform(sample: Sample) -> Sample:
        if preprocessor is not None:
            sample = preprocessor(sample)

        transformed = dict(**sample)
        transformed[image_key] = []
        transformed[objects_key] = []

        for idx in range(len(sample[image_key])):
            args: Sample = dict(
                image=img_to_np(sample[image_key][idx]),
                bboxes=sample[objects_key][idx][bbox_key],
            )
            # Include any "label" fields (albumentations will drop entries for
            # any boxes that get cut)
            for label_field in kwargs.get("label_fields", []):
                args[label_field] = sample[objects_key][idx][label_field]

            # Perform transform on the image+boxes
            res = img_bbox_transform(**args)

            transformed[image_key].append(img_from_np(res["image"]))

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
