import pytest
import torch
import torch.testing

from armory.metrics.detection import ObjectDetectionRates

pytestmark = pytest.mark.unit


@pytest.fixture
def target():
    return [
        {
            "labels": torch.tensor([2, 7, 6]),
            "boxes": torch.tensor(
                [
                    [0.1, 0.1, 0.7, 0.7],
                    [0.3, 0.3, 0.4, 0.4],
                    [0.05, 0.05, 0.15, 0.15],
                ]
            ),
        },
        {
            "labels": torch.tensor([]),
            "boxes": torch.tensor([]),
        },
        {
            "labels": torch.tensor([3]),
            "boxes": torch.tensor([[0.2, 0.2, 0.8, 0.8]]),
        },
    ]


@pytest.fixture
def preds():
    return [
        {
            "labels": torch.tensor([2, 9, 3]),
            "boxes": torch.tensor(
                [
                    [0.12, 0.09, 0.68, 0.7],
                    [0.5, 0.4, 0.9, 0.9],
                    [0.05, 0.05, 0.15, 0.15],
                ]
            ),
            "scores": torch.tensor([0.8, 0.8, 0.8]),
        },
        {
            "labels": torch.tensor([2]),
            "boxes": torch.tensor([[0.2, 0.2, 0.8, 0.8]]),
            "scores": torch.tensor([0.8]),
        },
        {
            "labels": torch.tensor([]),
            "boxes": torch.tensor([]),
            "scores": torch.tensor([]),
        },
    ]


def test_object_detection_rates_per_img(preds, target):
    metric = ObjectDetectionRates(iou_threshold=0.5, score_threshold=0.5, mean=False)
    results = metric(preds, target)

    assert len(results) == 4
    torch.testing.assert_close(
        results["true_positive_rate_per_img"], torch.tensor([1 / 3, 0.0, 0.0])
    )
    torch.testing.assert_close(
        results["misclassification_rate_per_img"], torch.tensor([1 / 3, 0.0, 0.0])
    )
    torch.testing.assert_close(
        results["disappearance_rate_per_img"], torch.tensor([1 / 3, 1.0, 1.0])
    )
    torch.testing.assert_close(
        results["hallucinations_per_img"], torch.tensor([1, 1, 0])
    )


def test_object_detection_rates_mean(preds, target):
    metric = ObjectDetectionRates(iou_threshold=0.5, score_threshold=0.5, mean=True)
    results = metric(preds, target)

    assert len(results) == 4
    torch.testing.assert_close(results["true_positive_rate_mean"], torch.tensor(1 / 9))
    torch.testing.assert_close(
        results["misclassification_rate_mean"], torch.tensor(1 / 9)
    )
    torch.testing.assert_close(results["disappearance_rate_mean"], torch.tensor(7 / 9))
    torch.testing.assert_close(results["hallucinations_mean"], torch.tensor(2 / 3))
