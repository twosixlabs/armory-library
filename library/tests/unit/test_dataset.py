import numpy as np
from numpy.testing import assert_array_equal
import pytest
import torch

import armory.data
from armory.dataset import (
    ArmoryDataset,
    ImageClassificationDataLoader,
    ObjectDetectionDataLoader,
    TupleDataset,
)

pytestmark = pytest.mark.unit


def test_ArmoryDataset_with_custom_adapter():
    raw_dataset = [
        {"data": [1, 2, 3], "target": 4},
        {"data": [5, 6, 7], "target": 8},
    ]

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


def test_ImageClassificationDataLoader():
    dataset = [
        {"data": torch.Tensor([1, 2, 3]), "target": np.array([4])},
        {"data": torch.Tensor([5, 6, 7]), "target": np.array([8])},
    ]
    dataloader = ImageClassificationDataLoader(
        dataset,
        dim=armory.data.ImageDimensions.CHW,
        image_key="data",
        label_key="target",
        scale=armory.data.Scale(dtype=armory.data.DataType.UINT8, max=255),
        batch_size=2,
    )
    batch = next(iter(dataloader))
    assert isinstance(batch, armory.data.ImageClassificationBatch)

    assert_array_equal(
        batch.inputs.to_numpy(dtype=np.uint8),
        np.array([[1, 2, 3], [5, 6, 7]], dtype=np.uint8),
        strict=True,
    )
    assert_array_equal(
        batch.targets.to_numpy(dtype=np.uint8),
        np.array([[4], [8]], dtype=np.uint8),
        strict=True,
    )


def test_ObjectDetectionDataLoader():
    dataset = [
        {
            "data": torch.Tensor([1, 2, 3]),
            "objects": {"box": [[1, 2, 3, 4], [6, 7, 8, 9]], "cat": [5, 0]},
        },
        {
            "data": torch.Tensor([5, 7, 9]),
            "objects": {"box": [[2, 4, 6, 8]], "cat": [3]},
        },
    ]
    dataloader = ObjectDetectionDataLoader(
        dataset,
        boxes_key="box",
        dim=armory.data.ImageDimensions.CHW,
        format=armory.data.BBoxFormat.XYXY,
        image_key="data",
        labels_key="cat",
        objects_key="objects",
        scale=armory.data.Scale(dtype=armory.data.DataType.UINT8, max=255),
        batch_size=2,
    )
    batch = next(iter(dataloader))
    assert isinstance(batch, armory.data.ObjectDetectionBatch)

    assert_array_equal(
        batch.inputs.to_numpy(dtype=np.uint8),
        np.array([[1, 2, 3], [5, 7, 9]], dtype=np.uint8),
        strict=True,
    )

    targets = batch.targets.to_numpy(np.uint8)
    assert len(targets) == 2
    assert_array_equal(
        targets[0]["boxes"],
        np.array([[1, 2, 3, 4], [6, 7, 8, 9]], dtype=np.uint8),
        strict=True,
    )
    assert_array_equal(
        targets[1]["boxes"],
        np.array([[2, 4, 6, 8]], dtype=np.uint8),
        strict=True,
    )
    assert_array_equal(targets[0]["labels"], np.array([5, 0]))
    assert_array_equal(targets[1]["labels"], np.array([3]))
