import pytest
import torch

from armory.metrics.tide import TIDE

pytestmark = pytest.mark.unit


def assert_main_error_count(results, error_key):
    all_error_keys = ["Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"]
    for key in all_error_keys:
        assert key in results["errors"]["main"]["count"], f"{key} not in results"
        actual_count = results["errors"]["main"]["count"][key]
        expected_count = 0 if key != error_key else 1
        assert (
            actual_count == expected_count
        ), f"Expected {expected_count} {key} errors but got {actual_count}"


def test_tide_with_no_errors():
    target = [
        {
            "labels": torch.tensor([1]),
            "boxes": torch.tensor([[10, 10, 10, 10]]),
        }
    ]
    preds = [
        {
            "labels": torch.tensor([1]),
            "boxes": torch.tensor([[11, 10, 10, 10]]),
            "scores": torch.tensor([0.8]),
        }
    ]
    metric = TIDE()
    results = metric(preds, target)
    assert_main_error_count(results, None)


def test_tide_with_classification_error():
    target = [
        {
            "labels": torch.tensor([1]),
            "boxes": torch.tensor([[10, 10, 10, 10]]),
        }
    ]
    preds = [
        {
            "labels": torch.tensor([2]),
            "boxes": torch.tensor([[11, 10, 10, 10]]),
            "scores": torch.tensor([0.8]),
        }
    ]
    metric = TIDE()
    results = metric(preds, target)
    assert_main_error_count(results, "Cls")


def test_tide_with_location_error():
    target = [
        {
            "labels": torch.tensor([1]),
            "boxes": torch.tensor([[10, 10, 10, 10]]),
        }
    ]
    preds = [
        {
            "labels": torch.tensor([1]),
            "boxes": torch.tensor([[16, 10, 10, 10]]),
            "scores": torch.tensor([0.8]),
        }
    ]
    metric = TIDE()
    results = metric(preds, target)
    assert_main_error_count(results, "Loc")


def test_tide_with_both_classification_and_location_error():
    target = [
        {
            "labels": torch.tensor([1]),
            "boxes": torch.tensor([[10, 10, 10, 10]]),
        }
    ]
    preds = [
        {
            "labels": torch.tensor([1, 2]),
            "boxes": torch.tensor([[11, 10, 10, 10], [16, 10, 10, 10]]),
            "scores": torch.tensor([0.8, 0.8]),
        }
    ]
    metric = TIDE()
    results = metric(preds, target)
    assert_main_error_count(results, "Both")


def test_tide_with_duplicate_error():
    target = [
        {
            "labels": torch.tensor([1]),
            "boxes": torch.tensor([[10, 10, 10, 10]]),
        }
    ]
    preds = [
        {
            "labels": torch.tensor([1, 1]),
            "boxes": torch.tensor([[11, 10, 10, 10], [12, 10, 10, 10]]),
            "scores": torch.tensor([0.8, 0.8]),
        }
    ]
    metric = TIDE()
    results = metric(preds, target)
    assert_main_error_count(results, "Dupe")


def test_tide_with_background_error():
    target = [
        {
            "labels": torch.tensor([1]),
            "boxes": torch.tensor([[10, 10, 10, 10]]),
        }
    ]
    preds = [
        {
            "labels": torch.tensor([1, 1]),
            "boxes": torch.tensor([[11, 10, 10, 10], [19, 10, 10, 10]]),
            "scores": torch.tensor([0.8, 0.8]),
        }
    ]
    metric = TIDE()
    results = metric(preds, target)
    assert_main_error_count(results, "Bkg")


def test_tide_with_missed_error():
    target = [
        {
            "labels": torch.tensor([1, 2]),
            "boxes": torch.tensor([[10, 10, 10, 10], [25, 25, 10, 10]]),
        }
    ]
    preds = [
        {
            "labels": torch.tensor([1]),
            "boxes": torch.tensor([[11, 10, 10, 10]]),
            "scores": torch.tensor([0.8]),
        }
    ]
    metric = TIDE()
    results = metric(preds, target)
    assert_main_error_count(results, "Miss")
