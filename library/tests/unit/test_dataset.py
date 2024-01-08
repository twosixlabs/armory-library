import numpy as np
from numpy.testing import assert_array_equal
import pytest

from charmory.dataset import ArmoryDataLoader, ArmoryDataset, TupleDataset

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
        return dict(x=np.array([1, 2, 3]), y=np.array([4]))

    dataset = ArmoryDataset(raw_dataset, adapter)
    assert len(dataset) == 2

    sample = dataset[0]
    assert_array_equal(sample["x"], np.array([1, 2, 3]), strict=True)
    assert_array_equal(sample["y"], np.array([4]), strict=True)


def test_TupleDataset():
    raw_dataset = [
        ([1, 2, 3], 4),
        ([5, 6, 7], 8),
    ]

    dataset = TupleDataset(raw_dataset, x_key="data", y_key="target")
    assert len(dataset) == 2

    sample = dataset[1]
    assert sample["data"] == [5, 6, 7]
    assert sample["target"] == 8


def test_ArmoryDataLoader(raw_dataset):
    loader = ArmoryDataLoader(raw_dataset, batch_size=2)
    batch = next(iter(loader))

    assert_array_equal(batch["data"], np.array([[1, 2, 3], [5, 6, 7]]), strict=True)
    assert_array_equal(batch["target"], np.array([4, 8]), strict=True)
