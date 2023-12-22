import functools

import numpy as np
from numpy.testing import assert_allclose
import pytest
import torch

import charmory.data as data

pytestmark = pytest.mark.unit


def deterministic_ndarray(dim, *dims, seed=0):
    if not dims:
        return [(0.1 * (seed + i) % 10) for i in range(dim)]
    return [deterministic_ndarray(*dims, seed=seed + i) for i in range(dim)]


@pytest.mark.parametrize(
    "constructor,method,dest_type",
    [
        (np.array, data.BatchedImages.numpy, np.ndarray),
        (np.array, data.BatchedImages.torch, torch.Tensor),
        (torch.tensor, data.BatchedImages.numpy, np.ndarray),
        (torch.tensor, data.BatchedImages.torch, torch.Tensor),
    ],
)
def test_BatchedImages_no_alterations(constructor, method, dest_type):
    images = data.BatchedImages(
        images=constructor(deterministic_ndarray(5, 100, 100, 3)),
        dim=data.ImageDimensions.HWC,
        scale=data.Scale(dtype=data.DataType.FLOAT, max=1.0),
    )
    as_torch = method(images)
    assert isinstance(as_torch, dest_type)
    assert_allclose(as_torch, deterministic_ndarray(5, 100, 100, 3))


@pytest.mark.parametrize(
    "module,method",
    [
        (np, data.BatchedImages.numpy),
        (np, data.BatchedImages.torch),
        (torch, data.BatchedImages.numpy),
        (torch, data.BatchedImages.torch),
    ],
)
def test_BatchedImages_hwc_to_chw(module, method):
    as_hwc = module.zeros((5, 100, 100, 3))
    as_hwc[3][1][4][1] = 0.59
    images = data.BatchedImages(
        images=as_hwc,
        dim=data.ImageDimensions.HWC,
        scale=data.Scale(dtype=data.DataType.FLOAT, max=1.0),
    )
    as_chw = method(images, dim=data.ImageDimensions.CHW)
    assert as_chw.shape == (5, 3, 100, 100)
    assert as_chw[3][1][1][4] == pytest.approx(0.59)


@pytest.mark.parametrize(
    "module,method",
    [
        (np, data.BatchedImages.numpy),
        (np, data.BatchedImages.torch),
        (torch, data.BatchedImages.numpy),
        (torch, data.BatchedImages.torch),
    ],
)
def test_BatchedImages_chw_to_hwc(module, method):
    as_chw = module.zeros((5, 3, 100, 100))
    as_chw[3][1][41][59] = 0.26
    images = data.BatchedImages(
        images=as_chw,
        dim=data.ImageDimensions.CHW,
        scale=data.Scale(dtype=data.DataType.FLOAT, max=1.0),
    )
    as_hwc = method(images, dim=data.ImageDimensions.HWC)
    assert as_hwc.shape == (5, 100, 100, 3)
    assert as_hwc[3][41][59][1] == pytest.approx(0.26)


@pytest.mark.parametrize(
    "src_module,dest_module,method",
    [
        (np, np, data.BatchedImages.numpy),
        (np, torch, data.BatchedImages.torch),
        (torch, np, data.BatchedImages.numpy),
        (torch, torch, data.BatchedImages.torch),
    ],
)
def test_BatchedImages_float_to_uint8(src_module, dest_module, method):
    as_float = src_module.zeros((5, 3, 100, 100))
    as_float[3][1][4][1] = 0.59
    images = data.BatchedImages(
        images=as_float,
        dim=data.ImageDimensions.CHW,
        scale=data.Scale(dtype=data.DataType.FLOAT, max=1.0),
    )
    as_uint8 = method(
        images,
        scale=data.Scale(dtype=data.DataType.UINT8, max=255),
        dtype=dest_module.uint8,
    )
    assert as_uint8.dtype == dest_module.uint8
    assert as_uint8[3][1][4][1] == 150


@pytest.mark.parametrize(
    "src_module,dest_module,method",
    [
        (np, np, data.BatchedImages.numpy),
        (np, torch, data.BatchedImages.torch),
        (torch, np, data.BatchedImages.numpy),
        (torch, torch, data.BatchedImages.torch),
    ],
)
def test_BatchedImages_uint8_to_float(src_module, dest_module, method):
    as_uint8 = src_module.zeros((5, 3, 100, 100), dtype=src_module.uint8)
    as_uint8[3][1][4][1] = 59
    images = data.BatchedImages(
        images=as_uint8,
        dim=data.ImageDimensions.CHW,
        scale=data.Scale(dtype=data.DataType.UINT8, max=255),
    )
    as_float = method(
        images,
        scale=data.Scale(dtype=data.DataType.FLOAT, max=1.0),
        dtype=dest_module.float32,
    )
    assert as_float.dtype == dest_module.float32
    assert as_float[3][1][4][1] == pytest.approx(0.23, 0.01)


@pytest.mark.parametrize(
    "module,method",
    [
        (np, data.BatchedImages.numpy),
        (np, data.BatchedImages.torch),
        (torch, data.BatchedImages.numpy),
        (torch, data.BatchedImages.torch),
    ],
)
def test_BatchedImages_normalized_hwc_to_unnormalized_hwc(module, method):
    normalized = module.zeros((5, 100, 100, 3))
    normalized[3][1][4][0] = 0.18
    normalized[3][1][4][1] = -0.96
    normalized[3][1][4][2] = 0.56
    images = data.BatchedImages(
        images=normalized,
        dim=data.ImageDimensions.HWC,
        scale=data.Scale(
            dtype=data.DataType.FLOAT,
            max=1.0,
            mean=(0.5, 0.5, 0.25),
            std=(0.5, 0.25, 0.5),
        ),
    )
    unnormalized = method(images, scale=data.Scale(dtype=data.DataType.FLOAT, max=1.0))
    assert unnormalized.shape == (5, 100, 100, 3)
    assert unnormalized[3][0][0][0] == pytest.approx(0.5)
    assert unnormalized[3][0][0][1] == pytest.approx(0.5)
    assert unnormalized[3][0][0][2] == pytest.approx(0.25)
    assert unnormalized[3][1][4][0] == pytest.approx(0.59)
    assert unnormalized[3][1][4][1] == pytest.approx(0.26)
    assert unnormalized[3][1][4][2] == pytest.approx(0.53)


@pytest.mark.parametrize(
    "module,method",
    [
        (np, data.BatchedImages.numpy),
        (np, data.BatchedImages.torch),
        (torch, data.BatchedImages.numpy),
        (torch, data.BatchedImages.torch),
    ],
)
def test_BatchedImages_unnormalized_hwc_to_normalized_chw(module, method):
    normalized = module.zeros((5, 100, 100, 3))
    normalized[3][1][4][0] = 0.59
    normalized[3][1][4][1] = 0.26
    normalized[3][1][4][2] = 0.53
    images = data.BatchedImages(
        images=normalized,
        dim=data.ImageDimensions.HWC,
        scale=data.Scale(dtype=data.DataType.FLOAT, max=1.0),
    )
    unnormalized = method(
        images,
        dim=data.ImageDimensions.CHW,
        scale=data.Scale(
            dtype=data.DataType.FLOAT,
            max=1.0,
            mean=(0.5, 0.5, 0.25),
            std=(0.5, 0.25, 0.5),
        ),
    )
    assert unnormalized.shape == (5, 3, 100, 100)
    assert unnormalized[3][0][0][0] == pytest.approx(-1.0)
    assert unnormalized[3][1][0][0] == pytest.approx(-2.0)
    assert unnormalized[3][2][0][0] == pytest.approx(-0.5)
    assert unnormalized[3][0][1][4] == pytest.approx(0.18)
    assert unnormalized[3][1][1][4] == pytest.approx(-0.96)
    assert unnormalized[3][2][1][4] == pytest.approx(0.56)


@pytest.mark.parametrize(
    "module,method",
    [
        (np, data.BatchedImages.numpy),
        (np, data.BatchedImages.torch),
        (torch, data.BatchedImages.numpy),
        (torch, data.BatchedImages.torch),
    ],
)
def test_BatchedImages_normalized_chw_to_normalized_hwc(module, method):
    normalized = module.zeros((5, 3, 100, 100))
    normalized[3][0][1][4] = 0.18
    normalized[3][1][1][4] = -0.96
    normalized[3][2][1][4] = 0.56
    images = data.BatchedImages(
        images=normalized,
        dim=data.ImageDimensions.CHW,
        scale=data.Scale(
            dtype=data.DataType.FLOAT,
            max=1.0,
            mean=(0.5, 0.5, 0.25),
            std=(0.5, 0.25, 0.5),
        ),
    )
    unnormalized = method(
        images,
        dim=data.ImageDimensions.HWC,
        scale=data.Scale(
            dtype=data.DataType.FLOAT,
            max=1.0,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    )
    assert unnormalized.shape == (5, 100, 100, 3)
    assert unnormalized[3][0][0][0] == pytest.approx(0.066, abs=0.001)
    assert unnormalized[3][0][0][1] == pytest.approx(0.196, abs=0.001)
    assert unnormalized[3][0][0][2] == pytest.approx(-0.693, abs=0.001)
    assert unnormalized[3][1][4][0] == pytest.approx(0.459, abs=0.001)
    assert unnormalized[3][1][4][1] == pytest.approx(-0.875, abs=0.001)
    assert unnormalized[3][1][4][2] == pytest.approx(0.551, abs=0.001)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA availability")
@pytest.mark.parametrize(
    "constructor",
    [
        np.array,
        torch.tensor,
        functools.partial(torch.tensor, device=torch.device("cpu")),
        functools.partial(torch.tensor, device=torch.device("cuda")),
    ],
)
def test_BatchedImages_to_gpu(constructor):
    images = data.BatchedImages(
        images=constructor(deterministic_ndarray(5, 100, 100, 3)),
        dim=data.ImageDimensions.HWC,
        scale=data.Scale(dtype=data.DataType.FLOAT, max=1.0),
    )
    as_torch = images.torch(device=torch.device("cuda", index=0))
    assert as_torch.device == torch.device("cuda", index=0)
    assert_allclose(as_torch.cpu(), deterministic_ndarray(5, 100, 100, 3))
