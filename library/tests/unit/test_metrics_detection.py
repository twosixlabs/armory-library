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
            "scores": torch.tensor([0.8, 0.6, 0.7]),
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
        results["disappearance_rate_per_img"], torch.tensor([1 / 3, 0.0, 1.0])
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
    torch.testing.assert_close(results["disappearance_rate_mean"], torch.tensor(4 / 9))
    torch.testing.assert_close(results["hallucinations_mean"], torch.tensor(2 / 3))


@pytest.mark.parametrize(
    "iou_threshold,expected_tpr,expected_mr,expected_dr,expected_h",
    [
        (0.5, 1 / 3, 1 / 3, 1 / 3, 1),
        (0.1, 1 / 3, 2 / 3, 0.0, 0),
        (0.95, 0.0, 1 / 3, 2 / 3, 2),
    ],
)
def test_object_detection_rates_iou_threshold(
    iou_threshold, expected_tpr, expected_mr, expected_dr, expected_h, preds, target
):
    metric = ObjectDetectionRates(
        iou_threshold=iou_threshold, score_threshold=0.5, mean=False
    )
    # Only run on the first image, ignore the image with no targets and no
    # detections because the threshold is irrelevant
    results = metric([preds[0]], [target[0]])

    assert len(results) == 4
    torch.testing.assert_close(
        results["true_positive_rate_per_img"], torch.tensor(expected_tpr)
    )
    torch.testing.assert_close(
        results["misclassification_rate_per_img"], torch.tensor(expected_mr)
    )
    torch.testing.assert_close(
        results["disappearance_rate_per_img"], torch.tensor(expected_dr)
    )
    torch.testing.assert_close(
        results["hallucinations_per_img"], torch.tensor(expected_h)
    )


@pytest.mark.parametrize(
    "score_threshold,expected_tpr,expected_mr,expected_dr,expected_h",
    [
        (0.50, 1 / 3, 1 / 3, 1 / 3, 1),
        (0.65, 1 / 3, 1 / 3, 1 / 3, 0),
        (0.75, 1 / 3, 0 / 3, 2 / 3, 0),
        (0.85, 0 / 3, 0 / 3, 3 / 3, 0),
    ],
)
def test_object_detection_rates_score_threshold(
    score_threshold, expected_tpr, expected_mr, expected_dr, expected_h, preds, target
):
    metric = ObjectDetectionRates(
        iou_threshold=0.5, score_threshold=score_threshold, mean=False
    )
    # Only run on the first image, ignore the image with no targets and no
    # detections because the threshold is irrelevant
    results = metric([preds[0]], [target[0]])

    assert len(results) == 4
    torch.testing.assert_close(
        results["true_positive_rate_per_img"], torch.tensor(expected_tpr)
    )
    torch.testing.assert_close(
        results["misclassification_rate_per_img"], torch.tensor(expected_mr)
    )
    torch.testing.assert_close(
        results["disappearance_rate_per_img"], torch.tensor(expected_dr)
    )
    torch.testing.assert_close(
        results["hallucinations_per_img"], torch.tensor(expected_h)
    )


@pytest.mark.parametrize(
    "class_list,expected_tpr,expected_mr,expected_dr,expected_h",
    [
        (None, 1 / 3, 1 / 3, 1 / 3, 1),
        ((2, 6, 7), 1 / 3, 0, 2 / 3, 0),
        ((2, 6), 1 / 2, 0, 1 / 2, 0),
        ((2,), 1, 0, 0, 0),
        ((0,), 0, 0, 0, 0),
    ],
)
def test_object_detection_rates_class_list(
    class_list, expected_tpr, expected_mr, expected_dr, expected_h, preds, target
):
    metric = ObjectDetectionRates(
        class_list=class_list, iou_threshold=0.5, score_threshold=0.5, mean=False
    )
    # Only run on the first image, ignore the image with no targets and no
    # detections because the class list is irrelevant
    results = metric([preds[0]], [target[0]])

    assert len(results) == 4
    torch.testing.assert_close(
        results["true_positive_rate_per_img"],
        torch.tensor(expected_tpr, dtype=torch.float32),
    )
    torch.testing.assert_close(
        results["misclassification_rate_per_img"],
        torch.tensor(expected_mr, dtype=torch.float32),
    )
    torch.testing.assert_close(
        results["disappearance_rate_per_img"],
        torch.tensor(expected_dr, dtype=torch.float32),
    )
    torch.testing.assert_close(
        results["hallucinations_per_img"], torch.tensor(expected_h)
    )
