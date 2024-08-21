import functools

import numpy as np
from numpy.testing import assert_allclose
import pytest
import torch

import armory.data as data

pytestmark = pytest.mark.unit


def deterministic_ndarray(dim, *dims, seed=0):
    if not dims:
        return [(0.1 * (seed + i) % 10) for i in range(dim)]
    return [deterministic_ndarray(*dims, seed=seed + i) for i in range(dim)]


@pytest.mark.parametrize(
    "constructor,spec,dest_type",
    [
        (np.array, data.NumpySpec(), np.ndarray),
        (np.array, data.TorchSpec(), torch.Tensor),
        (torch.tensor, data.NumpySpec(), np.ndarray),
        (torch.tensor, data.TorchSpec(), torch.Tensor),
    ],
)
def test_Images_no_alterations(constructor, spec, dest_type):
    images = data.Images(
        images=constructor(deterministic_ndarray(5, 100, 100, 3)),
        spec=data.ImageSpec(
            dim=data.ImageDimensions.HWC,
            scale=data.Scale(dtype=data.DataType.FLOAT, max=1.0),
        ),
    )
    as_torch = images.get(spec)
    assert isinstance(as_torch, dest_type)
    assert_allclose(as_torch, deterministic_ndarray(5, 100, 100, 3))


@pytest.mark.parametrize(
    "module,spec_cls,dest_type",
    [
        (np, data.ImageSpec, np.ndarray),
        (np, data.NumpyImageSpec, np.ndarray),
        (np, data.TorchImageSpec, torch.Tensor),
        (torch, data.ImageSpec, torch.Tensor),
        (torch, data.NumpyImageSpec, np.ndarray),
        (torch, data.TorchImageSpec, torch.Tensor),
    ],
)
def test_Images_hwc_to_chw(module, spec_cls, dest_type):
    as_hwc = module.zeros((5, 100, 100, 3))
    as_hwc[3][1][4][1] = 0.59
    images = data.Images(
        images=as_hwc,
        spec=data.ImageSpec(
            dim=data.ImageDimensions.HWC,
            scale=data.Scale(dtype=data.DataType.FLOAT, max=1.0),
        ),
    )
    as_chw = images.get(spec_cls(dim=data.ImageDimensions.CHW, scale=images.spec.scale))
    assert isinstance(as_chw, dest_type)
    assert as_chw.shape == (5, 3, 100, 100)
    assert as_chw[3][1][1][4] == pytest.approx(0.59)


@pytest.mark.parametrize(
    "module,spec_cls,dest_type",
    [
        (np, data.ImageSpec, np.ndarray),
        (np, data.NumpyImageSpec, np.ndarray),
        (np, data.TorchImageSpec, torch.Tensor),
        (torch, data.ImageSpec, torch.Tensor),
        (torch, data.NumpyImageSpec, np.ndarray),
        (torch, data.TorchImageSpec, torch.Tensor),
    ],
)
def test_Images_chw_to_hwc(module, spec_cls, dest_type):
    as_chw = module.zeros((5, 3, 100, 100))
    as_chw[3][1][41][59] = 0.26
    images = data.Images(
        images=as_chw,
        spec=data.ImageSpec(
            dim=data.ImageDimensions.CHW,
            scale=data.Scale(dtype=data.DataType.FLOAT, max=1.0),
        ),
    )
    as_hwc = images.get(spec_cls(dim=data.ImageDimensions.HWC, scale=images.spec.scale))
    assert isinstance(as_hwc, dest_type)
    assert as_hwc.shape == (5, 100, 100, 3)
    assert as_hwc[3][41][59][1] == pytest.approx(0.26)


@pytest.mark.parametrize(
    "src_module,dest_module,spec_cls",
    [
        (np, np, data.NumpyImageSpec),
        (np, torch, data.TorchImageSpec),
        (torch, np, data.NumpyImageSpec),
        (torch, torch, data.TorchImageSpec),
    ],
)
def test_Images_float_to_uint8(src_module, dest_module, spec_cls):
    as_float = src_module.zeros((5, 3, 100, 100))
    as_float[3][1][4][1] = 0.59
    images = data.Images(
        images=as_float,
        spec=data.ImageSpec(
            dim=data.ImageDimensions.CHW,
            scale=data.Scale(dtype=data.DataType.FLOAT, max=1.0),
        ),
    )
    as_uint8 = images.get(
        spec_cls(
            dim=images.spec.dim,
            scale=data.Scale(dtype=data.DataType.UINT8, max=255),
            dtype=dest_module.uint8,
        )
    )
    assert as_uint8.dtype == dest_module.uint8
    assert as_uint8[3][1][4][1] == 150


@pytest.mark.parametrize(
    "src_module,dest_module,spec_cls",
    [
        (np, np, data.NumpyImageSpec),
        (np, torch, data.TorchImageSpec),
        (torch, np, data.NumpyImageSpec),
        (torch, torch, data.TorchImageSpec),
    ],
)
def test_Images_uint8_to_float(src_module, dest_module, spec_cls):
    as_uint8 = src_module.zeros((5, 3, 100, 100), dtype=src_module.uint8)
    as_uint8[3][1][4][1] = 59
    images = data.Images(
        images=as_uint8,
        spec=data.ImageSpec(
            dim=data.ImageDimensions.CHW,
            scale=data.Scale(dtype=data.DataType.UINT8, max=255),
        ),
    )
    as_float = images.get(
        spec_cls(
            dim=images.spec.dim,
            scale=data.Scale(dtype=data.DataType.FLOAT, max=1.0),
            dtype=dest_module.float32,
        )
    )
    assert as_float.dtype == dest_module.float32
    assert as_float[3][1][4][1] == pytest.approx(0.23, 0.01)


@pytest.mark.parametrize(
    "module,spec_cls",
    [
        (np, data.ImageSpec),
        (np, data.NumpyImageSpec),
        (np, data.TorchImageSpec),
        (torch, data.ImageSpec),
        (torch, data.NumpyImageSpec),
        (torch, data.TorchImageSpec),
    ],
)
def test_Images_normalized_hwc_to_unnormalized_hwc(module, spec_cls):
    normalized = module.zeros((5, 100, 100, 3))
    normalized[3][1][4][0] = 0.18
    normalized[3][1][4][1] = -0.96
    normalized[3][1][4][2] = 0.56
    images = data.Images(
        images=normalized,
        spec=data.ImageSpec(
            dim=data.ImageDimensions.HWC,
            scale=data.Scale(
                dtype=data.DataType.FLOAT,
                max=1.0,
                mean=(0.5, 0.5, 0.25),
                std=(0.5, 0.25, 0.5),
            ),
        ),
    )
    unnormalized = images.get(
        spec_cls(
            dim=images.spec.dim, scale=data.Scale(dtype=data.DataType.FLOAT, max=1.0)
        )
    )
    assert unnormalized.shape == (5, 100, 100, 3)
    assert unnormalized[3][0][0][0] == pytest.approx(0.5)
    assert unnormalized[3][0][0][1] == pytest.approx(0.5)
    assert unnormalized[3][0][0][2] == pytest.approx(0.25)
    assert unnormalized[3][1][4][0] == pytest.approx(0.59)
    assert unnormalized[3][1][4][1] == pytest.approx(0.26)
    assert unnormalized[3][1][4][2] == pytest.approx(0.53)


@pytest.mark.parametrize(
    "module,spec_cls",
    [
        (np, data.ImageSpec),
        (np, data.NumpyImageSpec),
        (np, data.TorchImageSpec),
        (torch, data.ImageSpec),
        (torch, data.NumpyImageSpec),
        (torch, data.TorchImageSpec),
    ],
)
def test_Images_unnormalized_hwc_to_normalized_chw(module, spec_cls):
    normalized = module.zeros((5, 100, 100, 3))
    normalized[3][1][4][0] = 0.59
    normalized[3][1][4][1] = 0.26
    normalized[3][1][4][2] = 0.53
    images = data.Images(
        images=normalized,
        spec=data.ImageSpec(
            dim=data.ImageDimensions.HWC,
            scale=data.Scale(dtype=data.DataType.FLOAT, max=1.0),
        ),
    )
    unnormalized = images.get(
        spec_cls(
            dim=data.ImageDimensions.CHW,
            scale=data.Scale(
                dtype=data.DataType.FLOAT,
                max=1.0,
                mean=(0.5, 0.5, 0.25),
                std=(0.5, 0.25, 0.5),
            ),
        )
    )
    assert unnormalized.shape == (5, 3, 100, 100)
    assert unnormalized[3][0][0][0] == pytest.approx(-1.0)
    assert unnormalized[3][1][0][0] == pytest.approx(-2.0)
    assert unnormalized[3][2][0][0] == pytest.approx(-0.5)
    assert unnormalized[3][0][1][4] == pytest.approx(0.18)
    assert unnormalized[3][1][1][4] == pytest.approx(-0.96)
    assert unnormalized[3][2][1][4] == pytest.approx(0.56)


@pytest.mark.parametrize(
    "module,spec_cls",
    [
        (np, data.ImageSpec),
        (np, data.NumpyImageSpec),
        (np, data.TorchImageSpec),
        (torch, data.ImageSpec),
        (torch, data.NumpyImageSpec),
        (torch, data.TorchImageSpec),
    ],
)
def test_Images_normalized_chw_to_normalized_hwc(module, spec_cls):
    normalized = module.zeros((5, 3, 100, 100))
    normalized[3][0][1][4] = 0.18
    normalized[3][1][1][4] = -0.96
    normalized[3][2][1][4] = 0.56
    images = data.Images(
        images=normalized,
        spec=data.ImageSpec(
            dim=data.ImageDimensions.CHW,
            scale=data.Scale(
                dtype=data.DataType.FLOAT,
                max=1.0,
                mean=(0.5, 0.5, 0.25),
                std=(0.5, 0.25, 0.5),
            ),
        ),
    )
    unnormalized = images.get(
        spec_cls(
            dim=data.ImageDimensions.HWC,
            scale=data.Scale(
                dtype=data.DataType.FLOAT,
                max=1.0,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )
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
def test_Images_to_gpu(constructor):
    images = data.Images(
        images=constructor(deterministic_ndarray(5, 100, 100, 3)),
        spec=data.ImageSpec(
            dim=data.ImageDimensions.HWC,
            scale=data.Scale(dtype=data.DataType.FLOAT, max=1.0),
        ),
    )
    as_torch = images.get(data.TorchSpec(device=torch.device("cuda", index=0)))
    assert as_torch.device == torch.device("cuda", index=0)
    assert_allclose(as_torch.cpu(), deterministic_ndarray(5, 100, 100, 3))
