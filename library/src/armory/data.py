"""Armory data types"""

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial, singledispatch
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
# Debug/repr
###


@singledispatch
def debug(arg) -> str:
    return repr(arg)


@debug.register
def _(arg: list):
    return (
        f"[{debug(arg[0])}], ..., {debug(arg[-1])}]"
        if len(arg) > 2
        else "[" + ", ".join([debug(i) for i in arg]) + "]"
    )


@debug.register
def _(arg: dict):
    return "{" + ", ".join([f"'{k}': {debug(v)}" for k, v in arg.items()]) + "}"


@debug.register
def _(arg: np.ndarray):
    return f"<numpy.ndarray: shape={arg.shape} dtype={arg.dtype}>"


@debug.register
def _(arg: torch.Tensor):
    return f"<torch.Tensor: shape={arg.shape} dtype={arg.dtype}>"


###
# to_numpy
###


@singledispatch
def to_numpy(arg) -> np.ndarray:
    raise ValueError(f"Unsupported data type: {type(arg)}")


@to_numpy.register
def _(arg: np.ndarray):
    return arg


@to_numpy.register
def _(arg: torch.Tensor):
    return arg.cpu().numpy()


###
# to_torch
###


@singledispatch
def to_torch(arg) -> torch.Tensor:
    raise ValueError(f"Unsupported data type: {type(arg)}")


@to_torch.register
def _(arg: np.ndarray):
    return torch.from_numpy(arg)


@to_torch.register
def _(arg: torch.Tensor):
    return arg


@to_torch.register
def _(arg: list):
    return torch.as_tensor(arg)


###
# (torch) to_device
###


def to_device(arg: torch.Tensor, device: Optional[torch.device]):
    if device is not None and device != arg.device:
        return arg.to(device=device)
    return arg


###
# Image dimensions
###


class ImageDimensions(Enum):
    CHW = auto()
    HWC = auto()


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


@singledispatch
def convert_dim(
    data, from_dim: ImageDimensions, to_dim: Optional[ImageDimensions] = None
):
    raise ValueError(f"Unsupported data type: {type(data)}")


@convert_dim.register
def _(data: np.ndarray, from_dim, to_dim) -> np.ndarray:
    return _transpose(data, from_dim, to_dim, np.transpose)


@convert_dim.register
def _(data: torch.Tensor, from_dim, to_dim) -> torch.Tensor:
    return _transpose(data, from_dim, to_dim, torch.permute)


###
# Normalization
###


@singledispatch
def _copy(data):
    raise ValueError(f"Unsupported data type: {type(data)}")


@_copy.register
def _(data: np.ndarray) -> np.ndarray:
    return data.copy()


@_copy.register
def _(data: torch.Tensor) -> torch.Tensor:
    return data.clone()


def normalize(data, mean, std):
    normalized = _copy(data)
    for image in normalized:
        for c, m, s in zip(image, mean, std):
            c -= m
            c /= s
    return normalized


def unnormalize(data, mean, std):
    unnormalized = _copy(data)
    for image in unnormalized:
        for c, m, s in zip(image, mean, std):
            c *= s
            c += m
    return unnormalized


###
# Data types
###


@singledispatch
def to_float(arg):
    raise ValueError(f"Unsupported data type: {type(arg)}")


@to_float.register
def _(arg: np.ndarray):
    return arg.astype(dtype=np.float32)


@to_float.register
def _(arg: torch.Tensor):
    return arg.to(dtype=torch.float32)


@singledispatch
def to_dtype(arg, dtype):
    raise ValueError(f"Unsupported data type: {type(arg)}")


@to_dtype.register
def _(arg: np.ndarray, dtype: Optional[npt.DTypeLike]):
    if dtype is not None and dtype != arg.dtype:
        return arg.astype(dtype=dtype)
    return arg


@to_dtype.register
def _(arg: torch.Tensor, dtype: Optional[torch.dtype]):
    if dtype is not None and dtype != arg.dtype:
        return arg.to(dtype=dtype)
    return arg


###
# Scaling
###


class DataType(Enum):
    UINT8 = auto()
    FLOAT = auto()


@dataclass
class Scale:
    dtype: DataType
    max: Union[int, float]
    mean: Optional[Tuple[float, ...]] = None
    std: Optional[Tuple[float, ...]] = None

    @property
    def is_normalized(self) -> bool:
        return self.mean is not None and self.std is not None


def convert_scale(data, from_scale: Scale, to_scale: Optional[Scale] = None):
    if to_scale is None or to_scale == from_scale:
        return data

    data = to_float(data)

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
    XYXY = auto()
    XYWH = auto()
    CXCYWH = auto()


def to_bbox_format(
    data, from_format: BBoxFormat, to_format: Optional[BBoxFormat] = None
):
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
    def update(self, data: Any) -> None:
        ...


class SupportsMutation(SupportsConversion, SupportsUpdate, Protocol):
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
    def to(
        self,
        device: Optional[torch.device] = None,
    ):
        ...


class Metadata(TypedDict):
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
Batch = _Batch[SupportsConversion, SupportsConversion, SupportsMutation]


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
    RepresentationTypes = Union[np.ndarray, torch.Tensor]
    Accessor = Union[_Accessor[np.ndarray], _Accessor[torch.Tensor]]

    def __init__(
        self,
        images: RepresentationTypes,
        dim: ImageDimensions,
        scale: Scale,
    ):
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
        images = self._convert_dim_and_scale(self.images, dim, scale)
        images = to_numpy(images)
        images = to_dtype(images, dtype)
        return images

    def to_numpy(
        self,
        dtype: Optional[npt.DTypeLike] = None,
    ) -> np.ndarray:
        return self.to_numpy_images(dtype=dtype)

    @classmethod
    def as_torch(
        cls,
        dim: Optional[ImageDimensions] = None,
        scale: Optional[Scale] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> _Accessor[torch.Tensor]:
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
        return self.to_torch_images(dtype=dtype, device=device)


class NDimArray(SupportsMutation):
    Accessor = Union[_Accessor[np.ndarray], _Accessor[torch.Tensor]]

    def __init__(
        self,
        contents: Union[np.ndarray, torch.Tensor],
    ):
        self.contents = contents

    def __repr__(self) -> str:
        return f"NDimArray({debug(self.contents)})"

    def clone(self):
        return NDimArray(self.contents)

    def update(self, contents: Union[np.ndarray, torch.Tensor]):
        self.contents = contents

    @classmethod
    def as_numpy(
        cls,
        dtype: Optional[npt.DTypeLike] = None,
    ) -> _Accessor[np.ndarray]:
        return _CallableAccessor(
            get=partial(cls.to_numpy, dtype=dtype),
            set=partial(cls.update),
        )

    def to_numpy(
        self,
        dtype: Optional[npt.DTypeLike] = None,
    ) -> np.ndarray:
        return to_dtype(to_numpy(self.contents), dtype)

    @classmethod
    def as_torch(
        cls,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> _Accessor[torch.Tensor]:
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
        return to_dtype(to_device(to_torch(self.contents), device), dtype)


class BoundingBoxes(SupportsMutation):
    class BoxesNumpy(TypedDict):
        boxes: np.ndarray
        labels: np.ndarray
        scores: Optional[np.ndarray]

    class BoxesTorch(TypedDict):
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
        return self.to_torch_boxes(box_dtype=dtype, device=device)


###
# Batch types
###


class ImageClassificationBatch(_Batch[Images, NDimArray, NDimArray]):
    def __init__(
        self,
        inputs: Images,
        targets: NDimArray,
        metadata: Metadata,
        predictions: Optional[NDimArray] = None,
    ):
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
    def __init__(
        self,
        inputs: Images,
        targets: BoundingBoxes,
        metadata: Metadata,
        predictions: Optional[BoundingBoxes] = None,
    ):
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
