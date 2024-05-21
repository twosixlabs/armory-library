"""Armory data types"""

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import numpy.typing as npt
import torch
from torchvision.ops import box_convert
from typing_extensions import Self

###
# Conversion/utility functions
###


def debug(arg) -> str:
    """
    Creates a string message describing the argument, useful for debug logging.
    """
    if isinstance(arg, np.ndarray):
        return f"<numpy.ndarray: shape={arg.shape} dtype={arg.dtype}>"
    if isinstance(arg, torch.Tensor):
        return f"<torch.Tensor: shape={arg.shape} dtype={arg.dtype}>"
    if isinstance(arg, dict):
        return "{" + ", ".join([f"'{k}': {debug(v)}" for k, v in arg.items()]) + "}"
    if isinstance(arg, list):
        return (
            f"[{debug(arg[0])}], ..., {debug(arg[-1])}]"
            if len(arg) > 2
            else "[" + ", ".join([debug(i) for i in arg]) + "]"
        )
    return repr(arg)


def to_numpy(arg) -> np.ndarray:
    """Converts the argument to a NumPy array."""
    if isinstance(arg, np.ndarray):
        return arg
    if isinstance(arg, torch.Tensor):
        return arg.cpu().numpy()
    raise ValueError(f"Unsupported data type: {type(arg)}")


def to_torch(arg) -> torch.Tensor:
    """Converts the argument to a PyTorch tensor."""
    if isinstance(arg, np.ndarray):
        return torch.from_numpy(arg)
    if isinstance(arg, torch.Tensor):
        return arg
    if isinstance(arg, list):
        return torch.as_tensor(arg)
    raise ValueError(f"Unsupported data type: {type(arg)}")


def to_device(arg: torch.Tensor, device: Optional[torch.device]):
    """
    Moves the given PyTorch tensor to the given device, if one is specified and
    does not already match the tensor's current device.
    """
    if device is not None and device != arg.device:
        return arg.to(device=device)
    return arg


###
# Image dimensions
###


class ImageDimensions(Enum):
    """The format of a 3-dimensional data array containing image data."""

    CHW = auto()
    """
    Images whose n-dim array shape is (C, H, W) with the first dimension being
    the channels and the last two are the height and width.
    """
    HWC = auto()
    """
    Images whose n-dim array shape is (H, W, C) with the last dimension being
    the channels and the first two are the height and width.
    """


def _transpose(
    data, from_dim: ImageDimensions, to_dim: Optional[ImageDimensions], transpose
):
    if to_dim is None or to_dim == from_dim:
        return data
    if to_dim == ImageDimensions.CHW and from_dim == ImageDimensions.HWC:
        return transpose(data, (0, 3, 1, 2))
    if to_dim == ImageDimensions.HWC and from_dim == ImageDimensions.CHW:
        return transpose(data, (0, 2, 3, 1))
    raise ValueError(
        f"Invalid image dimension requested: requested {to_dim} from images with {from_dim}"
    )


def convert_dim(
    data, from_dim: ImageDimensions, to_dim: Optional[ImageDimensions] = None
):
    """
    Converts image data from the original dimension format to the new dimension
    format, if one is specified and does not already match the image data's
    current dimension format.
    """
    if isinstance(data, np.ndarray):
        return _transpose(data, from_dim, to_dim, np.transpose)
    if isinstance(data, torch.Tensor):
        return _transpose(data, from_dim, to_dim, torch.permute)
    raise ValueError(f"Unsupported data type: {type(data)}")


###
# Normalization
###


def _copy(data):
    if isinstance(data, np.ndarray):
        return data.copy()
    if isinstance(data, torch.Tensor):
        return data.clone()
    raise ValueError(f"Unsupported data type: {type(data)}")


def normalize(data, mean, std):
    """
    Normalizes the given image data using the given mean and standard deviation.
    """
    normalized = _copy(data)
    for image in normalized:
        for c, m, s in zip(image, mean, std):
            c -= m
            c /= s
    return normalized


def unnormalize(data, mean, std):
    """
    Unnormalizes the given normalized image data using the given mean and
    standard deviation.
    """
    unnormalized = _copy(data)
    for image in unnormalized:
        for c, m, s in zip(image, mean, std):
            c *= s
            c += m
    return unnormalized


###
# Data types
###

T = TypeVar("T")


def to_float_dtype(arg: T) -> T:
    """Converts the dtype of the argument to float."""
    if isinstance(arg, np.ndarray):
        return arg.astype(dtype=np.float32)
    if isinstance(arg, torch.Tensor):
        return arg.to(dtype=torch.float32)
    raise ValueError(f"Unsupported data type: {type(arg)}")


def to_dtype(arg: T, dtype) -> T:
    """Converts the dtype of the argument to the given dtype"""
    if isinstance(arg, np.ndarray):
        if dtype is not None and dtype != arg.dtype:
            return arg.astype(dtype=dtype)
        return arg
    if isinstance(arg, torch.Tensor):
        if dtype is not None and dtype != arg.dtype:
            return arg.to(dtype=dtype)
        return arg
    raise ValueError(f"Unsupported data type: {type(arg)}")


###
# Scaling
###


class DataType(Enum):
    """Data type for image data values."""

    UINT8 = auto()
    """Unsigned 8-bit integer."""
    FLOAT = auto()
    """Floating point."""


@dataclass
class Scale:
    """Image data scaling parameters."""

    dtype: DataType
    """The data type of the image data values."""
    max: Union[int, float]
    """The maximum value of the (unnormalized) image data values."""
    mean: Optional[Tuple[float, ...]] = None
    """If normalized, the mean used for normalization."""
    std: Optional[Tuple[float, ...]] = None
    """If normalized, the standard deviation used for normalization."""

    @property
    def is_normalized(self) -> bool:
        """Whether the image data has been normalized."""
        return self.mean is not None and self.std is not None


def convert_scale(data, from_scale: Scale, to_scale: Optional[Scale] = None):
    """
    Converts image data from the original scale to the new scale, if one is
    specified and does not already match the image data's current scale.
    """
    if to_scale is None or to_scale == from_scale:
        return data

    data = to_float_dtype(data)

    if from_scale.is_normalized:
        data = unnormalize(data, from_scale.mean, from_scale.std)

    if from_scale.max != to_scale.max:
        data = data * to_scale.max / from_scale.max

    if to_scale.is_normalized:
        data = normalize(data, to_scale.mean, to_scale.std)

    return data


###
# Bounding boxes
###


class BBoxFormat(Enum):
    """The format of bounding box coordinates."""

    XYXY = auto()
    """Coordinates are the upper left X and Y and lower right X and Y"""
    XYWH = auto()
    """Coordinates are the upper left X and Y and the width and height"""
    CXCYWH = auto()
    """Coordinates are the center X and Y and the widht and height"""


def to_bbox_format(
    data, from_format: BBoxFormat, to_format: Optional[BBoxFormat] = None
):
    """
    Converts bounding boxes from the original coordinate format to the new
    format, if one is specified and does not already match the bounding box's
    current coordinate format.
    """
    if len(data) == 0 or to_format is None or to_format == from_format:
        return data

    return box_convert(
        boxes=to_torch(data),
        in_fmt=from_format.name.lower(),
        out_fmt=to_format.name.lower(),
    )


###
# Protocols
###


class DataSpecification:
    """A specification for the structure, format, and values of raw data."""


class DataWithSpecification(Protocol):
    """
    Data with an accompanying specification, which may be converted to raw data
    of a different specification.
    """

    def __len__(self) -> int: ...

    def clone(self) -> Self: ...

    def get(self, specification: DataSpecification) -> object:
        """
        Retrieves a copy of the data with the given data specification. Specific
        subtypes will support different specifications according to the nature
        of the data.
        """
        ...

    def set(self, data: object, specification: DataSpecification) -> None:
        """Replaces the data with the given data specification."""
        ...


class Metadata(TypedDict):
    """
    Metadata about the source data or perturbations that have been applied to a
    batch.
    """

    data: Dict[str, Any]
    perturbations: Dict[str, Any]


class Batch(Protocol):
    """
    A collated sequence of samples to be processed simultaneously.
    """

    @property
    def initial_inputs(self) -> DataWithSpecification: ...

    @property
    def inputs(self) -> DataWithSpecification: ...

    @property
    def targets(self) -> DataWithSpecification: ...

    @property
    def metadata(self) -> Metadata: ...

    @property
    def predictions(self) -> DataWithSpecification: ...

    def clone(self): ...

    def __len__(self) -> int: ...


###
# Generic specification classes
###


@dataclass
class NumpySpec(DataSpecification):
    """A data specification for data types based on NumPy arrays."""

    dtype: Optional[npt.DTypeLike] = None
    """
    Optional, the NumPy dtype in which to represent data based on NumPy arrays.
    If none specified, the dtype will be unchanged.
    """


@dataclass
class TorchSpec(DataSpecification):
    """A data specification for data types based on PyTorch Tensors."""

    dtype: Optional[torch.dtype] = None
    """
    Optional, the PyTorch dtype in which to represent data based on PyTorch
    Tensors. If none specified, the dtype will be unchanged.
    """
    device: Optional[torch.device] = None
    """
    Optional, the PyTorch device on which to store the data. If none specified,
    the data will be unmoved from the originating device.
    """

    def __post_init__(self):
        # If this spec was created with an explicit target device, ignore
        # any future moves to a different device
        self._ignore_move = self.device is not None

    def to(self, device: torch.device) -> None:
        """
        Moves the target device of this spec to the given device, but only if
        the spec was not originally created with an explicit device.
        """
        if not self._ignore_move:
            self.device = device


###
# Convertable types
###


@dataclass
class ImageSpec(DataSpecification):

    dim: ImageDimensions
    """Image dimension format"""
    scale: Scale
    """Image data scale"""


@dataclass
class NumpyImageSpec(NumpySpec, ImageSpec):
    """Image data specification using NumPy arrays"""


@dataclass
class TorchImageSpec(TorchSpec, ImageSpec):
    """Image data specification using PyTorch Tensors"""


class Images(DataWithSpecification):
    """Computer vision model inputs"""

    _RawDataTypes = Union[np.ndarray, torch.Tensor]

    def __init__(
        self,
        images: _RawDataTypes,
        spec: ImageSpec,
    ):
        """
        Initializes the image data.

        Example::

            import torch
            from armory.data import DataType, ImageDimensions, ImageSpec, Images, Scale

            images = Images(
                images=torch.rand((3, 32, 32)),
                spec=ImageSpec(
                    dim=ImageDimensions.CHW,
                    scale=Scale(
                        dtype=DataType.FLOAT,
                        max=1.0,
                    ),
                ),
            )

        Args:
            images: Raw images data as either a NumPy array or a PyTorch Tensor
            spec: Specification for the structure, format, and values of the raw
                images data
        """
        self.images = images
        self.spec = spec

    def __repr__(self) -> str:
        return f"Images(images={debug(self.images)}, spec={self.spec})"

    def __len__(self) -> int:
        return self.images.shape[0]

    def clone(self):
        return Images(images=self.images, spec=self.spec)

    def _requires_renormalizing(self, to_spec: ImageSpec) -> bool:
        if to_spec.scale == self.spec.scale:
            return False  # no change to scaling
        if (
            to_spec.scale.mean == self.spec.scale.mean
            and to_spec.scale.std == self.spec.scale.std
        ):
            return False  # no change to normalization
        return True

    def _convert_to_image_spec(
        self, images: _RawDataTypes, to_spec: ImageSpec
    ) -> _RawDataTypes:
        from_dim = self.spec.dim

        if self._requires_renormalizing(to_spec):
            # Can only perform normalization of CHW
            images = convert_dim(images, from_dim, ImageDimensions.CHW)
            from_dim = ImageDimensions.CHW

        images = convert_scale(images, self.spec.scale, to_spec.scale)
        images = convert_dim(images, from_dim, to_spec.dim)
        return images

    @overload
    def get(self, spec: NumpySpec) -> np.ndarray: ...

    @overload
    def get(self, spec: TorchSpec) -> torch.Tensor: ...

    @overload
    def get(self, spec: ImageSpec) -> _RawDataTypes: ...

    def get(self, spec: Union[ImageSpec, NumpySpec, TorchSpec]) -> _RawDataTypes:
        """
        Retrieves a copy of the raw images data with the given image data
        specification.

        Example::

            from armory.data import DataType, ImageDimensions, NumpyImageSpec, Scale, TorchImageSpec

            # assuming `images` has been defined elsewhere
            images_pt = images.get(TorchImageSpec(
                dim=ImageDimensions.CHW,
                scale=Scale(
                    dtype=DataType.FLOAT,
                    max=1.0,
                ),
            ))

            images_np = images.get(NumpyImageSpec(
                dim=ImageDimensions.HWC,
                scale=Scale(
                    dtype=DataType.UINT8,
                    max=255,
                ),
            ))

        Args:
            spec: Specification for the structure, format, and values of the raw
                images data to be returned

        Return:
            Raw images data matching the requested specification
        """
        images = self.images
        # If requested type is torch, convert it and move to device first before
        # performing further operations
        if isinstance(spec, TorchSpec):
            images = to_torch(images)
            images = to_device(images, spec.device)
        if isinstance(spec, ImageSpec):
            images = self._convert_to_image_spec(images, spec)
        if isinstance(spec, NumpySpec):
            images = to_numpy(images)
            images = to_dtype(images, spec.dtype)
        if isinstance(spec, TorchSpec):
            # We do this down here even though we converted to torch above so
            # that any narrowing/widening effects due to data type conversion
            # occur after other operations
            images = to_dtype(images, spec.dtype)
        return images

    def set(self, images: _RawDataTypes, spec: ImageSpec) -> None:
        """
        Replaces the image data.

        Args:
            images: New raw images data
            spec: New image data specification
        """
        self.images = images
        if isinstance(spec, ImageSpec):
            # Don't update this if it's not an image spec (e.g., a generic
            # numpy/torch spec)
            self.spec = spec


class NDimArray(DataWithSpecification):
    """Variable-dimension data array"""

    _RawDataTypes = Union[np.ndarray, torch.Tensor]

    def __init__(self, contents: _RawDataTypes):
        """
        Initializes the data array.

        Args:
            contents: Raw data array
        """
        self.contents = contents

    def __repr__(self) -> str:
        return f"NDimArray({debug(self.contents)})"

    def __len__(self) -> int:
        return self.contents.shape[0]

    def clone(self):
        return NDimArray(self.contents)

    @overload
    def get(self, spec: NumpySpec) -> np.ndarray: ...

    @overload
    def get(self, spec: TorchSpec) -> torch.Tensor: ...

    def get(self, spec: Union[NumpySpec, TorchSpec]) -> _RawDataTypes:
        """
        Retrieves a copy of the raw data array with the given data specification.

        Example::

            from armory.data import NumpySpec, TorchSpec

            # assuming `data` has been defined elsewhere
            data_pt = data.get(TorchSpec())

            data_np = data.get(NumpySpec())

        Args:
            spec: Specification for the data type of the raw data array to be
                returned

        Return:
            Raw data array matching the requested specification
        """
        contents = self.contents
        if isinstance(spec, NumpySpec):
            contents = to_numpy(contents)
            contents = to_dtype(contents, spec.dtype)
        if isinstance(spec, TorchSpec):
            contents = to_torch(contents)
            contents = to_device(contents, spec.device)
            contents = to_dtype(contents, spec.dtype)
        return contents

    def set(self, contents: _RawDataTypes) -> None:
        """
        Replaces the data array.

        Args:
            contents: New raw data array
        """
        self.contents = contents


@dataclass
class BoundingBoxSpec(DataSpecification):

    format: BBoxFormat
    """Bounding box coordinate format"""


@dataclass
class NumpyBoundingBoxSpec(NumpySpec, BoundingBoxSpec):
    """Bounding box data specification using NumPy arrays"""

    box_dtype: Optional[npt.DTypeLike] = None
    """
    Optional, the NumPy dtype in which to represent bounding box coordinates.
    If none specified, the dtype will be unchanged.
    """
    label_dtype: Optional[npt.DTypeLike] = None
    """
    Optional, the NumPy dtype in which to represent object labels. If none
    specified, the dtype will be unchanged.
    """
    score_dtype: Optional[npt.DTypeLike] = None
    """
    Optional, the NumPy dtype in which to represent prediction scores.  If none
    specified, the dtype will be unchanged.
    """


@dataclass
class TorchBoundingBoxSpec(TorchSpec, BoundingBoxSpec):
    """Bounding box data specification using PyTorch Tensors"""

    box_dtype: Optional[torch.dtype] = None
    """
    Optional, the PyTorch dtype in which to represent bounding box coordinates.
    If none specified, the dtype will be unchanged.
    """
    label_dtype: Optional[torch.dtype] = None
    """
    Optional, the PyTorch dtype in which to represent object labels. If none
    specified, the dtype will be unchanged.
    """
    score_dtype: Optional[torch.dtype] = None
    """
    Optional, the PyTorch dtype in which to represent prediction scores.  If none
    specified, the dtype will be unchanged.
    """


class BoundingBoxes(DataWithSpecification):
    """Object detection targets or predictions"""

    class BoxesNumpy(TypedDict):
        """NumPy representation of bounding boxes"""

        boxes: np.ndarray
        labels: np.ndarray
        scores: Optional[np.ndarray]

    class BoxesTorch(TypedDict):
        """PyTorch tensor representation of bounding boxes"""

        boxes: torch.Tensor
        labels: torch.Tensor
        scores: Optional[torch.Tensor]

    _RawDataTypes = Union[Sequence[BoxesNumpy], Sequence[BoxesTorch]]

    def __init__(
        self,
        boxes: _RawDataTypes,
        spec: BoundingBoxSpec,
    ):
        """
        Initializes the bounding boxes.

        Example::

            import torch
            from armory.data import BBoxFormat, BoundingBoxSpec, BoundingBoxes

            boxes = BoundingBoxes(
                boxes=[
                    {
                        "boxes": torch.rand((3, 4)) * 32,
                        "labels": torch.rand(3),
                    },
                    {
                        "boxes": torch.rand((3, 4)) * 32,
                        "labels": torch.rand(3),
                    },
                ],
                spec=BoundingBoxSpec(format=BBoxFormat.XYXY),
            )

        Args:
            boxes: Raw bounding boxes data
            spec: Specification for the structure, format, and values of the raw
                bounding boxes data
        """
        self.boxes = boxes
        self.spec = spec

    def __repr__(self) -> str:
        boxes = (
            f"[{debug(self.boxes[0])}, ..., {debug(self.boxes[-1])}]"
            if len(self.boxes) > 2
            else debug(self.boxes)
        )
        return f"BoundingBoxes(boxes={boxes}, spec={self.spec})"

    def __len__(self) -> int:
        return len(self.boxes)

    def clone(self):
        return BoundingBoxes(boxes=self.boxes, spec=self.spec)

    @overload
    def get(self, spec: NumpySpec) -> Sequence[BoxesNumpy]: ...

    @overload
    def get(self, spec: TorchSpec) -> Sequence[BoxesTorch]: ...

    @overload
    def get(self, spec: BoundingBoxSpec) -> _RawDataTypes: ...

    def get(self, spec: Union[BoundingBoxSpec, NumpySpec, TorchSpec]) -> _RawDataTypes:
        """
        Retrieves a copy of the raw bounding boxes data with the given bounding
        box data specification.

        Example::

            from armory.data import BBoxFormat, NumpyBoundingBoxSpec, TorchBoundingBoxSpec

            # assuming `boxes` has been defined elsewhere
            boxes_pt = boxes.get(TorchBoundingBoxSpec(
                format=BBoxFormat.XYWH,
            ))

            boxes_np = boxes.get(NumpyBoundingBoxSpec(
                format=BBoxFormat.CXCYWH,
            ))

        Args:
            spec: Specification for the structure, format, and values of the raw
                bounding boxes data to be returned

        Return:
            Raw bounding boxes data matching the requested specification
        """
        boxes = [x["boxes"] for x in self.boxes]
        labels = [x["labels"] for x in self.boxes]
        scores = [x.get("scores", None) for x in self.boxes]

        # If requested type is torch, convert and move to device first before
        # performing further operations
        if isinstance(spec, TorchSpec):
            boxes = [to_device(to_torch(x), spec.device) for x in boxes]
            labels = [to_device(to_torch(x), spec.device) for x in labels]
            scores = [
                to_device(to_torch(x), spec.device) if x is not None else None
                for x in scores
            ]

        if isinstance(spec, BoundingBoxSpec):
            boxes = [to_bbox_format(x, self.spec.format, spec.format) for x in boxes]

        if isinstance(spec, NumpySpec):
            box_dtype = spec.dtype
            label_dtype = None
            score_dtype = None
            if isinstance(spec, NumpyBoundingBoxSpec):
                box_dtype = spec.box_dtype if box_dtype is None else box_dtype
                label_dtype = spec.label_dtype
                score_dtype = spec.score_dtype

            boxes = [to_dtype(to_numpy(x), box_dtype) for x in boxes]
            labels = [to_dtype(to_numpy(x), label_dtype) for x in labels]
            scores = [
                to_dtype(to_numpy(x), score_dtype) if x is not None else None
                for x in scores
            ]

        if isinstance(spec, TorchSpec):
            box_dtype = spec.dtype
            label_dtype = None
            score_dtype = None
            if isinstance(spec, TorchBoundingBoxSpec):
                box_dtype = spec.box_dtype if box_dtype is None else box_dtype
                label_dtype = spec.label_dtype
                score_dtype = spec.score_dtype

            boxes = [to_dtype(x, box_dtype) for x in boxes]
            labels = [to_dtype(x, label_dtype) for x in labels]
            scores = [
                to_dtype(x, score_dtype) if x is not None else None for x in scores
            ]

        return [
            dict(
                boxes=box,
                labels=label,
                scores=score,
            )
            for box, label, score in zip(boxes, labels, scores)
        ]  # type: ignore
        # We have to ignore typing errors here because although _we_ know that
        # the boxes/labels/scores are either all NumPy arrays or all PyTorch
        # tensors, mypy/pyright does not

    def set(self, boxes: _RawDataTypes, spec: BoundingBoxSpec) -> None:
        """
        Replaces the bounding boxes data.

        Args:
            images: New raw bounding boxes data
            spec: New bounding box data specification
        """
        self.boxes = boxes
        if isinstance(spec, BoundingBoxSpec):
            # Don't update this if it's not a bounding box spec (e.g., a generic
            # numpy/torch spec)
            self.spec = spec


###
# Batch types
###


class ImageClassificationBatch(Batch):
    """A batch of images and classified label/category predictions"""

    def __init__(
        self,
        inputs: Images,
        targets: NDimArray,
        metadata: Optional[Metadata] = None,
        predictions: Optional[NDimArray] = None,
    ):
        """
        Initializes the batch.

        Args:
            inputs: Images to be classified
            targets: Ground truth labels/categories of each image in the batch
            metadata: Optional, additional metadata about the samples in the
                batch
            predictions: Optional, the predicted labels/categories of each image
                in the batch
        """
        self._initial_inputs = inputs.clone()
        self._inputs = inputs
        self._targets = targets
        self._metadata = (
            metadata
            if metadata is not None
            else Metadata(data=dict(), perturbations=dict())
        )
        self._predictions = (
            predictions if predictions is not None else NDimArray(np.array([]))
        )

    def __repr__(self) -> str:
        return f"ImageClassificationBatch(inputs={self.inputs}, targets={self.targets}, metadata={self.metadata}, predictions={self.predictions})"

    @property
    def initial_inputs(self) -> Images:
        return self._initial_inputs

    @property
    def inputs(self) -> Images:
        return self._inputs

    @property
    def targets(self) -> NDimArray:
        return self._targets

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @property
    def predictions(self) -> NDimArray:
        return self._predictions

    def clone(self) -> "ImageClassificationBatch":
        return ImageClassificationBatch(
            inputs=self._inputs.clone(),
            targets=self._targets.clone(),
            metadata=deepcopy(self._metadata),
            predictions=self._predictions.clone(),
        )

    def __len__(self) -> int:
        return len(self.inputs)


class ObjectDetectionBatch(Batch):
    """A batch of images and detected object bounding box predictions"""

    def __init__(
        self,
        inputs: Images,
        targets: BoundingBoxes,
        metadata: Optional[Metadata] = None,
        predictions: Optional[BoundingBoxes] = None,
    ):
        """
        Initializes the batch.

        Args:
            inputs: Images in which to detect objects
            targets: Ground truth objects in each image in the batch
            metadata: Optional, additional metadata about the samples in the
                batch
            predictions: Optional, the predicted object bounding boxes in each
                image in the batch
        """
        self._initial_inputs = inputs.clone()
        self._inputs = inputs
        self._targets = targets
        self._metadata = (
            metadata
            if metadata is not None
            else Metadata(data=dict(), perturbations=dict())
        )
        self._predictions = (
            predictions
            if predictions is not None
            else BoundingBoxes([], spec=targets.spec)
        )

    def __repr__(self) -> str:
        return f"ObjectDetectionBatch(inputs={self.inputs}, targets={self.targets}, metadata={self.metadata}, predictions={self.predictions})"

    @property
    def initial_inputs(self) -> Images:
        return self._initial_inputs

    @property
    def inputs(self) -> Images:
        return self._inputs

    @property
    def targets(self) -> BoundingBoxes:
        return self._targets

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @property
    def predictions(self) -> BoundingBoxes:
        return self._predictions

    def clone(self) -> "ObjectDetectionBatch":
        return ObjectDetectionBatch(
            inputs=self._inputs.clone(),
            targets=self._targets.clone(),
            metadata=deepcopy(self._metadata),
            predictions=self._predictions.clone(),
        )

    def __len__(self) -> int:
        return len(self.inputs)
