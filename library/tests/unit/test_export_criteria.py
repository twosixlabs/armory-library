from unittest.mock import MagicMock

import pytest
import torch

from armory.data import Batch
import armory.export.criteria as criteria

pytestmark = pytest.mark.unit


@pytest.fixture
def batch():
    return MagicMock(spec=Batch)


@pytest.mark.parametrize(
    "n,batch_idx,expected",
    [
        (0, 0, False),
        (0, 1, False),
        (0, 2, False),
        (1, 0, True),
        (1, 1, True),
        (1, 2, True),
        (2, 0, False),
        (2, 1, True),
        (2, 2, False),
        (2, 3, True),
    ],
)
def test_every_n_batches(n, batch_idx, batch, expected):
    assert criteria.every_n_batches(n)("", batch_idx, batch) == expected


@pytest.mark.parametrize(
    "n,batch_idx,expected",
    [
        (0, 0, False),
        (0, 1, False),
        (0, 2, False),
        (1, 0, True),
        (1, 1, False),
        (1, 2, False),
        (2, 0, True),
        (2, 1, True),
        (2, 2, False),
        (2, 3, False),
    ],
)
def test_first_n_batches(n, batch_idx, batch, expected):
    assert criteria.first_n_batches(n)("", batch_idx, batch) == expected


@pytest.mark.parametrize(
    "n,batch_idx,batch_size,expected",
    [
        (0, 0, 2, False),
        (0, 1, 2, False),
        (0, 2, 2, False),
        (1, 0, 2, [0, 1]),
        (1, 1, 2, [0, 1]),
        (2, 0, 2, [1]),
        (2, 1, 2, [1]),
        (2, 0, 4, [1, 3]),
        (2, 1, 4, [1, 3]),
    ],
)
def test_every_n_samples(n, batch_idx, batch_size, expected):
    batch = MagicMock()
    batch.__len__ = MagicMock(return_value=batch_size)
    assert criteria.every_n_samples(n)("", batch_idx, batch) == expected


@pytest.mark.parametrize(
    "n,batch_idx,batch_size,expected",
    [
        (0, 0, 2, False),
        (0, 1, 2, False),
        (0, 2, 2, False),
        (1, 0, 2, [0, 1]),
        (1, 1, 2, [0, 1]),
        (2, 0, 2, [1]),
        (2, 1, 2, [1]),
        (3, 0, 4, [2]),
        (3, 1, 4, [2]),
    ],
)
def test_every_n_samples_of_batch(n, batch_idx, batch_size, expected):
    batch = MagicMock()
    batch.__len__ = MagicMock(return_value=batch_size)
    assert criteria.every_n_samples_of_batch(n)("", batch_idx, batch) == expected


@pytest.mark.parametrize(
    "n,batch_idx,batch_size,expected",
    [
        (0, 0, 2, False),
        (0, 1, 2, False),
        (0, 2, 2, False),
        (1, 0, 2, [0]),
        (1, 1, 2, False),
        (2, 0, 2, [0, 1]),
        (2, 1, 2, False),
        (2, 0, 4, [0, 1]),
        (2, 1, 4, False),
    ],
)
def test_first_n_samples(n, batch_idx, batch_size, expected):
    batch = MagicMock()
    batch.__len__ = MagicMock(return_value=batch_size)
    assert criteria.first_n_samples(n)("", batch_idx, batch) == expected


@pytest.mark.parametrize(
    "n,batch_idx,batch_size,expected",
    [
        (0, 0, 2, False),
        (0, 1, 2, False),
        (0, 2, 2, False),
        (1, 0, 2, [0]),
        (1, 1, 2, [0]),
        (2, 0, 2, [0, 1]),
        (2, 1, 2, [0, 1]),
        (2, 0, 4, [0, 1]),
        (2, 1, 4, [0, 1]),
    ],
)
def test_first_n_samples_of_batch(n, batch_idx, batch_size, expected):
    batch = MagicMock()
    batch.__len__ = MagicMock(return_value=batch_size)
    assert criteria.first_n_samples_of_batch(n)("", batch_idx, batch) == expected


@pytest.mark.parametrize(
    "indices,batch_idx,batch_size,expected",
    [
        ([], 0, 2, False),
        ([], 1, 2, False),
        ([1, 2], 0, 2, [1]),
        ([1, 2], 1, 2, [0]),
        ([3], 0, 2, []),
        ([3], 1, 2, [1]),
    ],
)
def test_samples(indices, batch_idx, batch_size, expected):
    batch = MagicMock()
    batch.__len__ = MagicMock(return_value=batch_size)
    assert criteria.samples(indices)("", batch_idx, batch) == expected


@pytest.mark.parametrize(
    "n_batch,n_sample,batch_idx,batch_size,expected",
    [
        (0, 1, 0, 4, []),
        (1, 0, 0, 4, []),
        (1, 1, 0, 4, [0, 1, 2, 3]),
        (1, 1, 1, 4, [0, 1, 2, 3]),
        (2, 1, 0, 4, []),
        (2, 1, 1, 4, [0, 1, 2, 3]),
        (1, 2, 0, 4, [1, 3]),
        (1, 2, 1, 4, [1, 3]),
        (2, 2, 0, 4, []),
        (2, 2, 1, 4, [1, 3]),
        (2, 2, 2, 4, []),
        (2, 2, 3, 4, [1, 3]),
    ],
)
def test_all_criteria(n_batch, n_sample, batch_idx, batch_size, expected):
    batch = MagicMock()
    batch.__len__ = MagicMock(return_value=batch_size)
    assert criteria.all_criteria(
        criteria.every_n_batches(n_batch),
        criteria.every_n_samples_of_batch(n_sample),
    )("", batch_idx, batch) == set(expected)


@pytest.mark.parametrize(
    "n_batch,n_sample,batch_idx,batch_size,expected",
    [
        (0, 1, 0, 4, [0, 1, 2, 3]),
        (1, 0, 0, 4, [0, 1, 2, 3]),
        (1, 1, 0, 4, [0, 1, 2, 3]),
        (1, 1, 1, 4, [0, 1, 2, 3]),
        (2, 1, 0, 4, [0, 1, 2, 3]),
        (2, 1, 1, 4, [0, 1, 2, 3]),
        (1, 2, 0, 4, [0, 1, 2, 3]),
        (1, 2, 1, 4, [0, 1, 2, 3]),
        (2, 2, 0, 4, [1, 3]),
        (2, 2, 1, 4, [0, 1, 2, 3]),
        (2, 2, 2, 4, [1, 3]),
        (2, 2, 3, 4, [0, 1, 2, 3]),
        (2, 3, 0, 4, [2]),
        (2, 3, 1, 4, [0, 1, 2, 3]),
        (2, 3, 2, 4, [0, 3]),
        (2, 3, 3, 4, [0, 1, 2, 3]),
    ],
)
def test_any_criteria(n_batch, n_sample, batch_idx, batch_size, expected):
    batch = MagicMock()
    batch.__len__ = MagicMock(return_value=batch_size)
    assert criteria.any_criteria(
        criteria.every_n_batches(n_batch),
        criteria.every_n_samples(n_sample),
    )("", batch_idx, batch) == set(expected)


@pytest.mark.parametrize(
    "metric_value,threshold,expected",
    [
        (0.1, 0.5, True),
        (0.5, 0.5, False),
        (0.7, 0.5, False),
        (torch.tensor(0.3), 0.5, True),
        (torch.tensor(0.7), 0.5, False),
        (torch.tensor([0.1, 0.2, 0.3, 0.4]), 0.5, [0, 1, 2, 3]),
        (torch.tensor([0.1, 0.5, 0.3, 0.4]), 0.5, [0, 2, 3]),
        (torch.tensor([0.6, 0.5, 0.8, 0.7]), 0.5, []),
        (torch.tensor([0.3]), 0.5, [0]),
        (torch.tensor([0.7]), 0.5, []),
    ],
)
def test_when_metric_lt(metric_value, threshold, batch, expected):
    metric = MagicMock(return_value=metric_value)
    assert criteria.when_metric_lt(metric, threshold)("", 0, batch) == expected


@pytest.mark.parametrize(
    "metric_value,threshold,expected",
    [
        (0.1, 0.5, False),
        (0.5, 0.5, False),
        (0.7, 0.5, True),
        (torch.tensor(0.3), 0.5, False),
        (torch.tensor(0.7), 0.5, True),
        (torch.tensor([0.1, 0.2, 0.3, 0.4]), 0.5, []),
        (torch.tensor([0.1, 0.6, 0.7, 0.4]), 0.5, [1, 2]),
        (torch.tensor([0.6, 0.9, 0.8, 0.7]), 0.5, [0, 1, 2, 3]),
        (torch.tensor([0.3]), 0.5, []),
        (torch.tensor([0.7]), 0.5, [0]),
    ],
)
def test_when_metric_gt(metric_value, threshold, batch, expected):
    metric = MagicMock(return_value=metric_value)
    assert criteria.when_metric_gt(metric, threshold)("", 0, batch) == expected
