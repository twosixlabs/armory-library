import numpy as np
from numpy.testing import assert_array_equal
import pytest

from charmory.data import (
    ArmoryDataLoader,
    ArmoryDataset,
    JaticImageClassificationDataset,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def raw_dataset():
    return [
        {"data": [1, 2, 3], "target": 4},
        {"data": [5, 6, 7], "target": 8},
    ]


def test_ArmoryDataset_with_custom_adapter(raw_dataset):
    def adapter(data):
        assert data == {"data": [1, 2, 3], "target": 4}
        return np.array([1, 2, 3]), np.array([4])

    dataset = ArmoryDataset(raw_dataset, adapter)
    assert len(dataset) == 2

    x, y = dataset[0]
    assert_array_equal(x, np.array([1, 2, 3]), strict=True)
    assert_array_equal(y, np.array([4]), strict=True)


def test_JaticImageClassificationDataset():
    raw_dataset = [
        {"image": [1, 2, 3], "label": 4},
        {"image": [5, 6, 7], "label": 8},
    ]
    dataset = JaticImageClassificationDataset(raw_dataset)
    assert len(dataset) == 2

    x, y = dataset[1]
    assert_array_equal(x, np.array([5, 6, 7]), strict=True)
    assert_array_equal(y, np.array(8), strict=True)


def test_ArmoryDataLoader(raw_dataset):
    def adapter(data):
        return data["data"], data["target"]

    dataset = ArmoryDataset(raw_dataset, adapter)
    loader = ArmoryDataLoader(dataset, batch_size=2)
    x, y = next(iter(loader))

    assert_array_equal(x, np.array([[1, 2, 3], [5, 6, 7]]), strict=True)
    assert_array_equal(y, np.array([4, 8]), strict=True)
