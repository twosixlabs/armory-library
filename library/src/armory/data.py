"""Armory data types"""

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt
import torch
from torchvision.ops import box_convert

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


def to_float_dtype(arg):
    """Converts the dtype of the argument to float."""
    if isinstance(arg, np.ndarray):
        return arg.astype(dtype=np.float32)
    if isinstance(arg, torch.Tensor):
        return arg.to(dtype=torch.float32)
    raise ValueError(f"Unsupported data type: {type(arg)}")


def to_dtype(arg, dtype):
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
    if to_format is None or to_format == from_format:
        return data

    return box_convert(
        boxes=to_torch(data),
        in_fmt=from_format.name.lower(),
        out_fmt=to_format.name.lower(),
    )


###
# Protocols
###


class SupportsConversion(Protocol):
    """A type whose data can be converted to framework-specific representations"""

    def to_numpy(self, dtype: Optional[npt.DTypeLike] = None) -> Any:
        """
        Generates a NumPy-based representation of the data. Specific subtypes may
        support additional conversion arguments.
        """
        ...

    def to_torch(
        self, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None
    ) -> Any:
        """
        Generates a PyTorch-based representation of the data. Specific subtypes may
        support additional conversion arguments.
        """
        ...


@runtime_checkable
class SupportsUpdate(Protocol):
    """A type whose low-level data representation can be updated"""

    def update(self, data: Any) -> None:
        """Updates the value to that of the given low-level data representation"""
        ...


class SupportsMutation(SupportsConversion, SupportsUpdate, Protocol):
    """A type whose data can be converted and updated"""

    pass


RepresentationType = TypeVar("RepresentationType")


class _Accessor(Protocol, Generic[RepresentationType]):
    """
    A pre-configured means of accessing or mutating the representation data of
    a convertable type.
    """

    def get(self, convertable: SupportsConversion) -> RepresentationType:
        """Obtains the representation data from a convertable type"""
        ...

    def set(self, convertable: SupportsConversion, data: RepresentationType) -> None:
        """Replaces the representation data for a convertable type"""
        ...


@runtime_checkable
class TorchAccessor(Protocol):
    """A data accessor that can be moved to a torch device"""

    def to(
        self,
        device: Optional[torch.device] = None,
    ):
        """Moves the accessor to the given device, if one is specified"""
        ...


class Metadata(TypedDict):
    """
    Metadata about the source data or perturbations that have been applied to a
    batch.
    """

    data: Dict[str, Any]
    perturbations: Dict[str, Any]


InputsType = TypeVar("InputsType", bound=SupportsConversion, covariant=True)
TargetsType = TypeVar("TargetsType", bound=SupportsConversion, covariant=True)
PredictionsType = TypeVar("PredictionsType", bound=SupportsMutation, covariant=True)


class _Batch(Protocol, Generic[InputsType, TargetsType, PredictionsType]):
    @property
    def initial_inputs(self) -> InputsType:
        ...

    @property
    def inputs(self) -> InputsType:
        ...

    @property
    def targets(self) -> TargetsType:
        ...

    @property
    def metadata(self) -> Metadata:
        ...

    @property
    def predictions(self) -> PredictionsType:
        ...

    def clone(self):
        ...

    def __len__(self) -> int:
        ...


Accessor = _Accessor[RepresentationType]
"""
A pre-configured means of accessing or mutating the representation data of
a convertable type.
"""

Batch = _Batch[SupportsConversion, SupportsConversion, SupportsMutation]
"""
A collated sequence of samples to be processed simultaneously.
"""


###
# Accessor classes
###


class _CallableAccessor(_Accessor[RepresentationType]):
    def __init__(
        self,
        get: Callable[[Any], RepresentationType],
        set: Callable[[Any, RepresentationType], None],
    ):
        self._get = get
        self._set = set

    def get(self, convertable) -> RepresentationType:
        """Obtains the representation data from a convertable type"""
        return self._get(convertable)

    def set(self, convertable, data: RepresentationType) -> None:
        self._set(convertable, data)


class _TorchCallableAccessor(TorchAccessor, _Accessor[RepresentationType]):
    def __init__(
        self,
        get: Callable[..., RepresentationType],
        set: Callable[[Any, RepresentationType], None],
        device: Optional[torch.device] = None,
    ):
        self._get = get
        self._set = set
        self.device = device

    def to(
        self,
        device: Optional[torch.device] = None,
    ):
        if self.device is None and device is not None:
            self.device = device

    def get(self, convertable) -> RepresentationType:
        """Obtains the representation data from a convertable type"""
        return self._get(convertable, device=self.device)

    def set(self, convertable, data: RepresentationType) -> None:
        self._set(convertable, data)


class DefaultNumpyAccessor(_Accessor[RepresentationType]):
    """A generic accessor to retrieve NumPy representations of data"""

    def get(self, convertable) -> RepresentationType:
        return convertable.to_numpy()

    def set(self, convertable, data: RepresentationType):
        if isinstance(convertable, SupportsUpdate):
            convertable.update(data)
        else:
            raise NotImplementedError(
                "Cannot mutate type using the default torch accessor"
            )


class DefaultTorchAccessor(TorchAccessor, _Accessor[RepresentationType]):
    """A generic accessor to retrieve PyTorch Tensor representations of data"""

    def __init__(
        self,
        device: Optional[torch.device] = None,
    ):
        self.device = device

    def to(
        self,
        device: Optional[torch.device] = None,
    ):
        if device is not None:
            self.device = device

    def get(self, convertable) -> RepresentationType:
        return convertable.to_torch(device=self.device)

    def set(self, convertable, data: RepresentationType):
        if isinstance(convertable, SupportsUpdate):
            convertable.update(data)
        else:
            raise NotImplementedError(
                "Cannot mutate type using the default torch accessor"
            )


###
# Convertable types
###


class Images(SupportsConversion, SupportsUpdate):
    """Computer vision model inputs"""

    RepresentationTypes = Union[np.ndarray, torch.Tensor]
    Accessor = Union[_Accessor[np.ndarray], _Accessor[torch.Tensor]]

    def __init__(
        self,
        images: RepresentationTypes,
        dim: ImageDimensions,
        scale: Scale,
    ):
        """
        Initializes the image data.

        Args:
            images: Low-level representation of images
            dim: Image dimension format
            scale: Image data scale
        """
        self.images = images
        self.dim = dim
        self.scale = scale

    def __repr__(self) -> str:
        return f"Images(images={debug(self.images)}, dim={self.dim} scale={self.scale})"

    def __len__(self) -> int:
        return len(self.images)

    def clone(self):
        return Images(images=self.images, dim=self.dim, scale=self.scale)

    def update(
        self,
        images: RepresentationTypes,
        dim: Optional[ImageDimensions] = None,
        scale: Optional[Scale] = None,
    ):
        """
        Updates the image data.

        Args:
            images: New low-level representation of images
            dim: Optional, image dimension format if changed
            scale: Optional, image data scale if changed
        """
        self.images = images
        if dim is not None:
            self.dim = dim
        if scale is not None:
            self.scale = scale

    def _requires_renormalizing(self, to_scale: Optional[Scale]):
        if to_scale is None or to_scale == self.scale:
            return False  # no change to scaling
        if to_scale.mean == self.scale.mean and to_scale.std == self.scale.std:
            return False  # no change to normalization
        return True

    def _convert_dim_and_scale(
        self, images, dim: Optional[ImageDimensions], scale: Optional[Scale]
    ):
        from_dim = self.dim

        if self._requires_renormalizing(scale):
            # Can only perform normalization of CHW
            images = convert_dim(images, from_dim, ImageDimensions.CHW)
            if dim is None:
                dim = from_dim  # convert back to the original dimensions
            from_dim = ImageDimensions.CHW

        images = convert_scale(images, self.scale, scale)
        images = convert_dim(images, from_dim, dim)
        return images

    @classmethod
    def as_numpy(
        cls,
        dim: Optional[ImageDimensions] = None,
        scale: Optional[Scale] = None,
        dtype: Optional[npt.DTypeLike] = None,
    ) -> _Accessor[np.ndarray]:
        """
        Creates a NumPy accessor for images. The returned accessor will
        produce/accept NumPy arrays of the specified dimension and scale.

        Args:
            dim: Optional, the image dimension format in which to represent the
                images. If none specified, the dimensions will be unchanged from
                the current dimension format when images are accessed.
            scale: Optional, the image data scale in which to represent the
                images. If none specified, the data scale will be unchanged from
                the current data scale when images are accessed.
            dtype: Optional, the NumPy dtype in which to represent the images.
                If none specified, the dtype will be unchanged when images are
                accessed.

        Return:
            Data accessor instance
        """
        return _CallableAccessor(
            get=partial(cls.to_numpy_images, dim=dim, scale=scale, dtype=dtype),
            set=partial(cls.update, dim=dim, scale=scale),
        )

    def to_numpy_images(
        self,
        dim: Optional[ImageDimensions] = None,
        scale: Optional[Scale] = None,
        dtype: Optional[npt.DTypeLike] = None,
    ) -> np.ndarray:
        """
        Creates a NumPy array representation of the images.

        Args:
            dim: Optional, the image dimension format in which to represent the
                images. If none specified, the dimensions will be unchanged from
                the current dimension format.
            scale: Optional, the image data scale in which to represent the
                images. If none specified, the data scale will be unchanged from
                the current data scale.
            dtype: Optional, the NumPy dtype in which to represent the images.
                If none specified, the dtype will be unchanged.

        Return:
            NumPy array representation of the image data
        """
        images = self._convert_dim_and_scale(self.images, dim, scale)
        images = to_numpy(images)
        images = to_dtype(images, dtype)
        return images

    def to_numpy(
        self,
        dtype: Optional[npt.DTypeLike] = None,
    ) -> np.ndarray:
        """
        Creates a NumPy array representation of the images.

        Args:
            dtype: Optional, the NumPy dtype in which to represent the images.
                If none specified, the dtype will be unchanged.

        Return:
            NumPy array representation of the image data
        """
        return self.to_numpy_images(dtype=dtype)

    @classmethod
    def as_torch(
        cls,
        dim: Optional[ImageDimensions] = None,
        scale: Optional[Scale] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> _Accessor[torch.Tensor]:
        """
        Creates a PyTorch tensor accessor for images. The returned accessor will
        produce/accept PyTorch tensors of the specified dimension and scale.

        Args:
            dim: Optional, the image dimension format in which to represent the
                images. If none specified, the dimensions will be unchanged from
                the current dimension format when images are accessed.
            scale: Optional, the image data scale in which to represent the
                images. If none specified, the data scale will be unchanged from
                the current data scale when images are accessed.
            dtype: Optional, the PyTorch dtype in which to represent the images.
                If none specified, the dtype will be unchanged when images are
                accessed.
            device: Optional, the PyTorch device on which to represent the images.
                If none specified, the data will be unmoved from the originating
                device when images are accessed.

        Return:
            Data accessor instance
        """
        return _TorchCallableAccessor[torch.Tensor](
            get=partial(cls.to_torch_images, dim=dim, scale=scale, dtype=dtype),
            set=partial(cls.update, dim=dim, scale=scale),
            device=device,
        )

    def to_torch_images(
        self,
        dim: Optional[ImageDimensions] = None,
        scale: Optional[Scale] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Creates a PyTorch tensor representation of the images.

        Args:
            dim: Optional, the image dimension format in which to represent the
                images. If none specified, the dimensions will be unchanged from
                the current dimension format.
            scale: Optional, the image data scale in which to represent the
                images. If none specified, the data scale will be unchanged from
                the current data scale.
            dtype: Optional, the PyTorch dtype in which to represent the images.
                If none specified, the dtype will be unchanged.
            device: Optional, the PyTorch device on which to represent the images.
                If none specified, the data will be unmoved from the originating
                device.

        Return:
            PyTorch tensor representation of the image data
        """
        # Convert to torch and move to device first before performing conversions
        images = to_torch(self.images)
        images = to_device(images, device)
        images = self._convert_dim_and_scale(images, dim, scale)
        images = to_dtype(images, dtype)
        return images

    def to_torch(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Creates a PyTorch tensor representation of the images.

        Args:
            dtype: Optional, the PyTorch dtype in which to represent the images.
                If none specified, the dtype will be unchanged.
            device: Optional, the PyTorch device on which to represent the images.
                If none specified, the data will be unmoved from the originating
                device.

        Return:
            PyTorch tensor representation of the image data
        """
        return self.to_torch_images(dtype=dtype, device=device)


class NDimArray(SupportsMutation):
    """Variable-dimension data array"""

    Accessor = Union[_Accessor[np.ndarray], _Accessor[torch.Tensor]]

    def __init__(
        self,
        contents: Union[np.ndarray, torch.Tensor],
    ):
        """
        Initializes the data array.

        Args:
            contents: Low-level representation of data array
        """
        self.contents = contents

    def __repr__(self) -> str:
        return f"NDimArray({debug(self.contents)})"

    def clone(self):
        return NDimArray(self.contents)

    def update(self, contents: Union[np.ndarray, torch.Tensor]):
        """
        Updates the data array.

        Args:
            contents: New low-level representation of data array
        """
        self.contents = contents

    @classmethod
    def as_numpy(
        cls,
        dtype: Optional[npt.DTypeLike] = None,
    ) -> _Accessor[np.ndarray]:
        """
        Creates a NumPy accessor for the array. The returned accessor will
        produce/accept NumPy arrays.

        Args:
            dtype: Optional, the NumPy dtype in which to represent the array.
                If none specified, the dtype will be unchanged when arrays are
                accessed.

        Return:
            Data accessor instance
        """
        return _CallableAccessor(
            get=partial(cls.to_numpy, dtype=dtype),
            set=partial(cls.update),
        )

    def to_numpy(
        self,
        dtype: Optional[npt.DTypeLike] = None,
    ) -> np.ndarray:
        """
        Creates a NumPy array representation of the data array.

        Args:
            dtype: Optional, the NumPy dtype in which to represent the array.
                If none specified, the dtype will be unchanged.

        Return:
            NumPy array representation of the data array
        """
        return to_dtype(to_numpy(self.contents), dtype)

    @classmethod
    def as_torch(
        cls,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> _Accessor[torch.Tensor]:
        """
        Creates a PyTorch tensor accessor for the array. The returned accessor
        will produce/accept PyTorch tensors.

        Args:
            dtype: Optional, the PyTorch dtype in which to represent the array.
                If none specified, the dtype will be unchanged when arrays are
                accessed.
            device: Optional, the PyTorch device on which to represent the array.
                If none specified, the data will be unmoved from the originating
                device when arrays are accessed.

        Return:
            Data accessor instance
        """
        return _TorchCallableAccessor[torch.Tensor](
            get=partial(cls.to_torch, dtype=dtype),
            set=partial(cls.update),
            device=device,
        )

    def to_torch(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Creates a PyTorch tensor representation of the array.

        Args:
            dtype: Optional, the PyTorch dtype in which to represent the array.
                If none specified, the dtype will be unchanged.
            device: Optional, the PyTorch device on which to represent the array.
                If none specified, the data will be unmoved from the originating
                device.

        Return:
            PyTorch tensor representation of the data array
        """
        return to_dtype(to_device(to_torch(self.contents), device), dtype)


class BoundingBoxes(SupportsMutation):
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

    RepresentationTypes = Union[Sequence[BoxesNumpy], Sequence[BoxesTorch]]
    Accessor = Union[_Accessor[Sequence[BoxesNumpy]], _Accessor[Sequence[BoxesTorch]]]

    def __init__(
        self,
        boxes: RepresentationTypes,
        format: BBoxFormat,
    ):
        """
        Initializes the bounding boxes.

        Args:
            boxes: Low-level representation of bounding boxes
            format: Bounding box coordinate format
        """
        self.boxes = boxes
        self.format = format

    def __repr__(self) -> str:
        boxes = (
            f"[{debug(self.boxes[0])}, ..., {debug(self.boxes[-1])}]"
            if len(self.boxes) > 2
            else debug(self.boxes)
        )
        return f"BoundingBoxes(boxes={boxes}, format={self.format})"

    def __len__(self) -> int:
        return len(self.boxes)

    def clone(self):
        return BoundingBoxes(boxes=self.boxes, format=self.format)

    def update(
        self,
        boxes: RepresentationTypes,
        format: Optional[BBoxFormat] = None,
    ):
        """
        Updates the bounding boxes.

        Args:
            boxes: New low-level representation of bounding boxes
            format: Optional, bounding box coordinate format if changed
        """
        self.boxes = boxes
        if format is not None:
            self.format = format

    @classmethod
    def as_numpy(
        cls,
        format: Optional[BBoxFormat] = None,
        box_dtype: Optional[npt.DTypeLike] = None,
        label_dtype: Optional[npt.DTypeLike] = None,
        score_dtype: Optional[npt.DTypeLike] = None,
    ) -> _Accessor[Sequence[BoxesNumpy]]:
        """
        Creates a NumPy accessor for bounding boxes. The returned accessor will
        produce/accept NumPy-based objects of the specified format.

        Args:
            format: Optional, the bounding box coordinate format in which to
                represent the bounding boxes. If none specified, the coordinate
                format will be unchanged from the current coordinate format when
                bounding boxes are accessed.
            box_dtype: Optional, the NumPy dtype in which to represent the
                bounding box coordinates. If none specified, the dtype will be
                unchanged when bounding boxes are accessed.
            label_dtype: Optional, the NumPy dtype in which to represent the
                object labels. If none specified, the dtype will be unchanged
                when bounding boxes are accessed.
            score_dtype: Optional, the NumPy dtype in which to represent the
                prediction scores. If none specified, the dtype will be
                unchanged when bounding boxes are accessed.

        Return:
            Data accessor instance
        """
        return _CallableAccessor(
            get=partial(
                cls.to_numpy_boxes,
                format=format,
                box_dtype=box_dtype,
                label_dtype=label_dtype,
                score_dtype=score_dtype,
            ),
            set=partial(cls.update, format=format),
        )

    def to_numpy_boxes(
        self,
        format: Optional[BBoxFormat] = None,
        box_dtype: Optional[npt.DTypeLike] = None,
        label_dtype: Optional[npt.DTypeLike] = None,
        score_dtype: Optional[npt.DTypeLike] = None,
    ) -> Sequence[BoxesNumpy]:
        """
        Creates a NumPy-based representation of the bounding boxes.

        Args:
            format: Optional, the bounding box coordinate format in which to
                represent the bounding boxes. If none specified, the coordinate
                format will be unchanged from the current coordinate format.
            box_dtype: Optional, the NumPy dtype in which to represent the
                bounding box coordinates. If none specified, the dtype will be
                unchanged.
            label_dtype: Optional, the NumPy dtype in which to represent the
                object labels. If none specified, the dtype will be unchanged.
            score_dtype: Optional, the NumPy dtype in which to represent the
                prediction scores. If none specified, the dtype will be
                unchanged.

        Return:
            NumPy-based representation of the bounding boxes
        """
        return [
            self.BoxesNumpy(
                boxes=to_dtype(
                    to_numpy(to_bbox_format(box["boxes"], self.format, format)),
                    box_dtype,
                ),
                labels=to_dtype(to_numpy(box["labels"]), label_dtype),
                scores=(
                    to_dtype(to_numpy(box["scores"]), score_dtype)
                    if box.get("scores", None) is not None
                    else None
                ),
            )
            for box in self.boxes
        ]

    def to_numpy(self, dtype: Optional[npt.DTypeLike] = None) -> Sequence[BoxesNumpy]:
        """
        Creates a NumPy-based representation of the bounding boxes.

        Args:
            dtype: Optional, the NumPy dtype in which to represent the
                bounding box coordinates. If none specified, the dtype will be
                unchanged.

        Return:
            NumPy-based representation of the bounding boxes
        """
        return self.to_numpy_boxes(box_dtype=dtype)

    @classmethod
    def as_torch(
        cls,
        format: Optional[BBoxFormat] = None,
        box_dtype: Optional[torch.dtype] = None,
        label_dtype: Optional[torch.dtype] = None,
        score_dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> _Accessor[Sequence[BoxesTorch]]:
        """
        Creates a PyTorch tensor accessor for bounding boxes. The returned
        accessor will produce/accept PyTorch tensor-based objects of the
        specified format.

        Args:
            format: Optional, the bounding box coordinate format in which to
                represent the bounding boxes. If none specified, the coordinate
                format will be unchanged from the current coordinate format when
                bounding boxes are accessed.
            box_dtype: Optional, the PyTorch dtype in which to represent the
                bounding box coordinates. If none specified, the dtype will be
                unchanged when bounding boxes are accessed.
            label_dtype: Optional, the PyTorch dtype in which to represent the
                object labels. If none specified, the dtype will be unchanged
                when bounding boxes are accessed.
            score_dtype: Optional, the PyTorch dtype in which to represent the
                prediction scores. If none specified, the dtype will be
                unchanged when bounding boxes are accessed.
            device: Optional, the PyTorch device on which to represent the
                bounding boxes.  If none specified, the data will be unmoved
                from the originating device when images are accessed.

        Return:
            Data accessor instance
        """
        return _TorchCallableAccessor(
            get=partial(
                cls.to_torch_boxes,
                format=format,
                box_dtype=box_dtype,
                label_dtype=label_dtype,
                score_dtype=score_dtype,
                device=device,
            ),
            set=partial(cls.update, format=format),
        )

    def to_torch_boxes(
        self,
        format: Optional[BBoxFormat] = None,
        box_dtype: Optional[torch.dtype] = None,
        label_dtype: Optional[torch.dtype] = None,
        score_dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Sequence[BoxesTorch]:
        """
        Creates a PyTorch tensor-based representation of the bounding boxes.

        Args:
            format: Optional, the bounding box coordinate format in which to
                represent the bounding boxes. If none specified, the coordinate
                format will be unchanged from the current coordinate format.
            box_dtype: Optional, the PyTorch dtype in which to represent the
                bounding box coordinates. If none specified, the dtype will be
                unchanged.
            label_dtype: Optional, the PyTorch dtype in which to represent the
                object labels. If none specified, the dtype will be unchanged.
            score_dtype: Optional, the PyTorch dtype in which to represent the
                prediction scores. If none specified, the dtype will be
                unchanged.
            device: Optional, the PyTorch device on which to represent the
                bounding boxes.  If none specified, the data will be unmoved
                from the originating device.

        Return:
            PyTorch tensor-based representation of the bounding boxes
        """
        return [
            self.BoxesTorch(
                boxes=to_dtype(
                    to_bbox_format(
                        to_device(to_torch(box["boxes"]), device), self.format, format
                    ),
                    box_dtype,
                ),
                labels=to_dtype(
                    to_device(to_torch(box["labels"]), device), label_dtype
                ),
                scores=(
                    to_dtype(to_device(to_torch(box["scores"]), device), score_dtype)
                    if box.get("scores", None) is not None
                    else None
                ),
            )
            for box in self.boxes
        ]

    def to_torch(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Sequence[BoxesTorch]:
        """
        Creates a PyTorch tensor-based representation of the bounding boxes.

        Args:
            dtype: Optional, the PyTorch dtype in which to represent the
                bounding box coordinates. If none specified, the dtype will be
                unchanged.
            device: Optional, the PyTorch device on which to represent the
                bounding boxes.  If none specified, the data will be unmoved
                from the originating device.

        Return:
            PyTorch tensor-based representation of the bounding boxes
        """
        return self.to_torch_boxes(box_dtype=dtype, device=device)


###
# Batch types
###


class ImageClassificationBatch(_Batch[Images, NDimArray, NDimArray]):
    """A batch of images and classified label/category predictions"""

    def __init__(
        self,
        inputs: Images,
        targets: NDimArray,
        metadata: Metadata,
        predictions: Optional[NDimArray] = None,
    ):
        """
        Initializes the batch.

        Args:
            inputs: Images to be classified
            targets: Ground truth labels/categories of each image in the batch
            metadata: Additional metadata about the samples in the batch
            predictions: Optional, the predicted labels/categories of each image
                in the batch
        """
        self._initial_inputs = inputs.clone()
        self._inputs = inputs
        self._targets = targets
        self._metadata = metadata
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


class ObjectDetectionBatch(_Batch[Images, BoundingBoxes, BoundingBoxes]):
    """A batch of images and detected object bounding box predictions"""

    def __init__(
        self,
        inputs: Images,
        targets: BoundingBoxes,
        metadata: Metadata,
        predictions: Optional[BoundingBoxes] = None,
    ):
        """
        Initializes the batch.

        Args:
            inputs: Images in which to detect objects
            targets: Ground truth objects in each image in the batch
            metadata: Additional metadata about the samples in the batch
            predictions: Optional, the predicted object bounding boxes in each
                image in the batch
        """
        self._initial_inputs = inputs.clone()
        self._inputs = inputs
        self._targets = targets
        self._metadata = metadata
        self._predictions = (
            predictions
            if predictions is not None
            else BoundingBoxes([], format=targets.format)
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
