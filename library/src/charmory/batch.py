"""Armory batch objects"""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum, auto
from functools import partial, singledispatch
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Protocol,
    Self,
    Sequence,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import numpy.typing as npt
import torch
from torchvision.ops import box_convert

#
# torch vs numpy
# images: HWC vs CHW
# uint8 (0-255) vs float (0.0-1.0) (vs unbound float?)
# normalized (-3.0-3.0) vs unnormalized
# tensor: gpu vs cpu?
# boxes: XYXY vs XYWH vs CXCYWH
#
#


def is_supported_type(arg):
    if isinstance(arg, np.ndarray):
        return True
    if isinstance(arg, torch.Tensor):
        return True
    return False


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


class ImageDimensions(StrEnum):
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


class DataType(StrEnum):
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
# Protocols
###


class SupportsConversion(Protocol):
    """A type whose data can be converted to framework-specific representations"""

    def numpy(self, dtype: Optional[npt.DTypeLike] = None) -> Any:
        """
        Generates a NumPy-based representation of the data. Specific subtypes may
        support additional conversion arguments.
        """
        ...

    def torch(
        self, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None
    ) -> Any:
        """
        Generates a PyTorch-based representation of the data. Specific subtypes may
        support additional conversion arguments.
        """
        ...


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


InputsType = TypeVar("InputsType", bound=SupportsConversion, covariant=True)
TargetsType = TypeVar("TargetsType", bound=SupportsConversion, covariant=True)
PredictionsType = TypeVar("PredictionsType", bound=SupportsConversion, covariant=True)


class _Batch(Protocol, Generic[InputsType, TargetsType, PredictionsType]):
    initial_inputs: InputsType
    inputs: InputsType
    targets: TargetsType
    metadata: Dict[str, Any]
    predictions: Optional[PredictionsType]

    def clone(self) -> Self:
        ...

    # @property
    # def inputs(self) -> InputsType:
    #     ...

    # @inputs.setter
    # def inputs(self, inputs: InputsType):
    #     ...

    # @property
    # def targets(self) -> SupportsConversion:
    #     ...

    # @property
    # def metadata(self) -> Dict[str, Any]:
    #     ...

    # @property
    # def predictions(self) -> Optional[SupportsConversion]:
    #     ...

    # @predictions.setter
    # def predictions(self, predictions: SupportsConversion):
    #     ...


Accessor = _Accessor[RepresentationType]
Batch = _Batch[SupportsConversion, SupportsConversion, SupportsConversion]


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


InputType = TypeVar("InputType")
T = TypeVar("T")


class BatchedImages(SupportsConversion):
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

    def clone(self):
        return BatchedImages(images=self.images, dim=self.dim, scale=self.scale)

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

    def numpy(
        self,
        dim: Optional[ImageDimensions] = None,
        scale: Optional[Scale] = None,
        dtype: Optional[npt.DTypeLike] = None,
    ) -> np.ndarray:
        images = self._convert_dim_and_scale(self.images, dim, scale)
        images = to_numpy(images)
        images = to_dtype(images, dtype)
        return images

    @classmethod
    def as_numpy(
        cls,
        dim: Optional[ImageDimensions] = None,
        scale: Optional[Scale] = None,
        dtype: Optional[npt.DTypeLike] = None,
    ) -> _Accessor[np.ndarray]:
        return _CallableAccessor(
            get=partial(cls.numpy, dim=dim, scale=scale, dtype=dtype),
            set=partial(cls.update, dim=dim, scale=scale),
        )

    @classmethod
    def as_torch(
        cls,
        dim: Optional[ImageDimensions] = None,
        scale: Optional[Scale] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> _Accessor[torch.Tensor]:
        return _CallableAccessor(
            get=partial(cls.torch, dim=dim, scale=scale, dtype=dtype, device=device),
            set=partial(cls.update, dim=dim, scale=scale),
        )

    def torch(
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


class NDimArray(SupportsConversion):
    def __init__(
        self,
        contents: Union[np.ndarray, torch.Tensor],
    ):
        self.contents = contents

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
            get=partial(cls.numpy, dtype=dtype),
            set=partial(cls.update),
        )

    @classmethod
    def as_torch(
        cls,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> _Accessor[torch.Tensor]:
        return _CallableAccessor(
            get=partial(cls.torch, dtype=dtype, device=device),
            set=partial(cls.update),
        )

    def numpy(
        self,
        dtype: Optional[npt.DTypeLike] = None,
    ) -> np.ndarray:
        return to_dtype(to_numpy(self.contents), dtype)

    def torch(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        return to_dtype(to_device(to_torch(self.contents), device), dtype)


class Batch2(ABC):
    @abstractmethod
    def get_inputs_numpy(self):
        ...

    @abstractmethod
    def get_inputs_torch(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        ...

    @abstractmethod
    def replace_inputs(self, inputs):
        ...

    @abstractmethod
    def get_targets_numpy(self):
        ...

    @abstractmethod
    def get_targets_torch(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        ...


class ComputerVisionBatch(Batch2):
    class Dimensions(StrEnum):
        CHW = auto()
        HWC = auto()

    def __init__(self, inputs: Union[np.ndarray, torch.Tensor], dim: Dimensions):
        self._inputs = inputs
        self.dim = dim

    def _transpose(self, data, dim: Optional[Dimensions], transpose):
        if dim is None or dim == self.dim:
            return data
        if dim == self.Dimensions.CHW and self.dim == self.Dimensions.HWC:
            return transpose(data, (0, 3, 1, 2))
        if dim == self.Dimensions.HWC and self.dim == self.Dimensions.CHW:
            return transpose(data, (0, 2, 3, 1))
        raise ValueError(
            f"Invalid image dimension requested: requested {dim} from batch with {self.dim}"
        )

    def get_inputs_numpy(self, dim: Optional[Dimensions] = None) -> np.ndarray:
        if isinstance(self._inputs, np.ndarray):
            inputs = self._inputs
        elif isinstance(self._inputs, torch.Tensor):
            inputs = self._inputs.cpu().numpy()
        else:
            raise ValueError(f"Invalid type of batch inputs: {type(self._inputs)}")

        return self._transpose(inputs, dim, np.transpose)

    def get_inputs_torch(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        dim: Optional[Dimensions] = None,
    ) -> torch.Tensor:
        if isinstance(self._inputs, np.ndarray):
            inputs = torch.from_numpy(self._inputs)
        elif isinstance(self._inputs, torch.Tensor):
            inputs = self._inputs
        else:
            raise ValueError(f"Invalid type of batch inputs: {type(self._inputs)}")

        if (dtype is not None and dtype != inputs.dtype) or (
            device is not None and device != inputs.device
        ):
            inputs = inputs.to(dtype=dtype, device=device)

        return self._transpose(inputs, dim, torch.permute)

    def replace_inputs(
        self,
        inputs: Union[np.ndarray, torch.Tensor],
        dim: Optional[Dimensions] = None,
    ):
        self._inputs = inputs
        if dim is not None:
            self.dim = dim


class ImageClassificationBatch(_Batch[BatchedImages, NDimArray, NDimArray]):
    def __init__(
        self,
        inputs: BatchedImages,
        targets: NDimArray,
        metadata: Dict[str, Any],
    ):
        self.initial_inputs = inputs.clone()
        self.inputs = inputs
        self.targets = targets
        self.metadata = metadata
        self.predictions: Optional[NDimArray] = None

    def clone(self) -> "ImageClassificationBatch":
        copy = ImageClassificationBatch(
            inputs=self.inputs.clone(),
            targets=self.targets.clone(),
            metadata=deepcopy(self.metadata),
        )
        copy.predictions = self.predictions
        return copy

    # @property
    # def inputs(self) -> BatchedImages:
    #     return self._inputs

    # @property
    # def targets(self) -> NDimArray:
    #     return self._targets

    # @property
    # def metadata(self) -> Dict[str, Any]:
    #     return self._metadata


# def get_targets_numpy(self) -> np.ndarray:
#     if isinstance(self._targets, np.ndarray):
#         return self._targets
#     elif isinstance(self._targets, torch.Tensor):
#         return self._targets.cpu().numpy()
#     else:
#         raise ValueError(f"Invalid type of batch targets: {type(self._targets)}")

# def get_targets_torch(
#     self,
#     dtype: Optional[torch.dtype] = None,
#     device: Optional[torch.device] = None,
# ) -> torch.Tensor:
#     if isinstance(self._targets, np.ndarray):
#         targets = torch.from_numpy(self._targets)
#     elif isinstance(self._targets, torch.Tensor):
#         targets = self._targets
#     else:
#         raise ValueError(f"Invalid type of batch targets: {type(self._targets)}")

#     if (dtype is not None and dtype != targets.dtype) or (
#         device is not None and device != targets.device
#     ):
#         targets = targets.to(dtype=dtype, device=device)

#     return targets

# def add_predictions(self, preds: Union[np.ndarray, torch.Tensor]) -> None:
#     self._preds = preds

# def get_predictions_numpy(self) -> np.ndarray:
#     if isinstance(self._preds, np.ndarray):
#         return self._preds
#     elif isinstance(self._preds, torch.Tensor):
#         return self._preds.cpu().numpy()
#     else:
#         raise ValueError(f"Invalid type of batch predictions: {type(self._preds)}")

# def get_predictions_torch(
#     self,
#     dtype: Optional[torch.dtype] = None,
#     device: Optional[torch.device] = None,
# ) -> torch.Tensor:
#     if isinstance(self._preds, np.ndarray):
#         preds = torch.from_numpy(self._preds)
#     elif isinstance(self._preds, torch.Tensor):
#         preds = self._preds
#     else:
#         raise ValueError(f"Invalid type of batch predictions: {type(self._preds)}")

#     if (dtype is not None and dtype != preds.dtype) or (
#         device is not None and device != preds.device
#     ):
#         preds = preds.to(dtype=dtype, device=device)

#     return preds


class ObjectDetectionBatch(ComputerVisionBatch):
    class BBoxFormat(StrEnum):
        XYXY = auto()
        XYWH = auto()
        CXCYWH = auto()

    class ObjectsNumpy(TypedDict):
        boxes: np.ndarray
        labels: np.ndarray

    class ObjectsTorch(TypedDict):
        boxes: torch.FloatTensor
        labels: torch.IntTensor

    class PredictionsNumpy(ObjectsNumpy):
        scores: np.ndarray

    class PredictionsTorch(ObjectsTorch):
        scores: torch.FloatTensor

    def __init__(
        self,
        inputs: Union[np.ndarray, torch.Tensor],
        targets: Union[Sequence[ObjectsNumpy], Sequence[ObjectsTorch]],
        dim: ComputerVisionBatch.Dimensions,
        bboxformat: BBoxFormat,
    ):
        super().__init__(inputs=inputs, dim=dim)
        self._targets = targets
        self.targets_bboxformat = bboxformat
        self._preds: Union[
            None,
            Sequence[ObjectDetectionBatch.PredictionsNumpy],
            Sequence[ObjectDetectionBatch.PredictionsTorch],
        ] = None
        self.preds_bboxformat = bboxformat

    def get_targets_numpy(
        self,
        bboxformat: Optional[BBoxFormat] = None,
    ) -> Sequence[ObjectsNumpy]:
        if len(self._targets) == 0:
            return []

        if isinstance(self._targets[0]["boxes"], np.ndarray):
            if bboxformat is None or bboxformat == self.targets_bboxformat:
                return cast(Sequence[ObjectDetectionBatch.ObjectsNumpy], self._targets)
            return [
                self.ObjectsNumpy(
                    boxes=box_convert(
                        torch.tensor(target["boxes"]),
                        self.targets_bboxformat,
                        bboxformat,
                    ).numpy(),
                    labels=target["labels"],
                )
                for target in cast(
                    Sequence[ObjectDetectionBatch.ObjectsNumpy], self._targets
                )
            ]
        elif isinstance(self._targets[0]["boxes"], torch.Tensor):
            return [
                self.ObjectsNumpy(
                    boxes=target["boxes"].cpu().numpy(),
                    labels=target["labels"].cpu().numpy(),
                )
                for target in cast(
                    Sequence[ObjectDetectionBatch.ObjectsTorch], self._targets
                )
            ]
        else:
            raise ValueError(
                f"Invalid type of batch targets: {type(self._targets[0]['boxes'])}"
            )

    def get_targets_torch(
        self,
        device: Optional[torch.device] = None,
    ) -> Sequence[ObjectsTorch]:
        if len(self._targets) == 0:
            return []

        if isinstance(self._targets[0]["boxes"], np.ndarray):
            return [
                self.ObjectsTorch(
                    boxes=torch.FloatTensor(target["boxes"], device=device),
                    labels=torch.IntTensor(target["labels"], device=device),
                )
                for target in cast(
                    Sequence[ObjectDetectionBatch.ObjectsNumpy], self._targets
                )
            ]
        else:
            raise ValueError(
                f"Invalid type of batch targets: {type(self._targets[0]['boxes'])}"
            )

    def add_predictions(
        self,
        preds: Union[Sequence[PredictionsNumpy,], Sequence[PredictionsTorch]],
        bboxformat: BBoxFormat,
    ) -> None:
        ...

    def get_predictions_numpy(self) -> Sequence[PredictionsNumpy]:
        ...

    def get_predictions_torch(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        bboxformat: Optional[BBoxFormat] = None,
    ) -> Sequence[PredictionsTorch]:
        ...
