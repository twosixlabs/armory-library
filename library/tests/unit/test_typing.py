import numpy as np
import pytest
import torch

import charmory.typing as typing

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "value",
    [
        np.array([1, 2, 3]),
        torch.as_tensor([1, 2, 3]),
        [1, 2, 3],
    ],
)
def test_coerce_to_np(value):
    assert isinstance(typing.coerce(value, np.ndarray), np.ndarray)


@pytest.mark.parametrize(
    "value",
    [
        np.array([1, 2, 3]),
        torch.as_tensor([1, 2, 3]),
        [1, 2, 3],
    ],
)
def test_coerce_to_torch(value):
    assert isinstance(typing.coerce(value, torch.Tensor), torch.Tensor)


def test_autocoerce_positional_only():
    @typing.autocoerce
    def func(x: torch.Tensor, /, **kwargs):
        assert isinstance(x, torch.Tensor)

    func(np.array([1, 2, 3]))
    with pytest.raises(TypeError):
        func(x=np.array([1, 2, 3]))


def test_autocoerce_positional_or_keyword():
    @typing.autocoerce
    def func(x: torch.Tensor):
        assert isinstance(x, torch.Tensor)

    func(np.array([1, 2, 3]))
    func(x=np.array([1, 2, 3]))


def test_autocoerce_keyword_only():
    @typing.autocoerce
    def func(*, x: torch.Tensor):
        assert isinstance(x, torch.Tensor)

    with pytest.raises(TypeError):
        func(np.array([1, 2, 3]))
    func(x=np.array([1, 2, 3]))
