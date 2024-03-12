"""Object detection metrics"""

from typing import List, Optional, Sequence

import torch
from torchmetrics import Metric
from torchmetrics.functional.detection import intersection_over_union
from torchmetrics.utilities import dim_zero_cat

from armory.data import BBoxFormat, BoundingBoxes
from armory.metric import PredictionMetric


class ObjectDetectionRates(Metric):
    """
    Metric for the following object detection metrics:

    - True positive rate: the percent of ground-truth boxes which are predicted
      with iou > iou_threshold, score > score_threshold, and the correct label
    - Misclassification rate: the percent of ground-truth boxes with are
      predicted with iou > iou_threshold, score > score_threshold, and the
      incorrect label
    - Disappearance rate: 1 - true positive rate - misclassification rate
    - Hallucinations: the number of predicted boxes per image that have
      score > score_threshold and iou < iou_threshold for each ground-truth box
    """

    @classmethod
    def create(cls, *args, **kwargs):
        """
        Creates an instance of the object detection rates metric pre-wrapped in
        a `armory.metric.PredictionMetric`
        """
        return PredictionMetric(
            cls(*args, **kwargs), BoundingBoxes.as_torch(format=BBoxFormat.XYXY)
        )

    def __init__(
        self,
        mean: bool = True,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.5,
        class_list: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.mean = mean
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.class_list = (
            torch.as_tensor(class_list) if class_list is not None else None
        )

        self.true_positive_rate_per_img: List[torch.Tensor]
        self.misclassification_rate_per_img: List[torch.Tensor]
        self.disappearance_rate_per_img: List[torch.Tensor]
        self.hallucinations_per_img: List[torch.Tensor]

        self.add_state("true_positive_rate_per_img", default=[], dist_reduce_fx="cat")
        self.add_state(
            "misclassification_rate_per_img", default=[], dist_reduce_fx="cat"
        )
        self.add_state("disappearance_rate_per_img", default=[], dist_reduce_fx="cat")
        self.add_state("hallucinations_per_img", default=[], dist_reduce_fx="cat")

    def update(
        self,
        preds: Sequence[BoundingBoxes.BoxesTorch],
        targets: Sequence[BoundingBoxes.BoxesTorch],
    ):
        for y, y_pred in zip(targets, preds):
            if self.class_list:
                # Filter out ground-truth classes with labels not in class_list
                indices_to_keep = torch.where(torch.isin(y["labels"], self.class_list))
                gt_boxes = y["boxes"][indices_to_keep]
                gt_labels = y["labels"][indices_to_keep]
            else:
                gt_boxes = y["boxes"]
                gt_labels = y["labels"]

            # initialize count of hallucinations
            num_hallucinations = torch.tensor(0)
            num_gt_boxes = len(gt_boxes)

            # Initialize arrays that will indicate whether each respective ground-truth
            # box is a true positive or misclassified
            true_positive_array = torch.zeros((num_gt_boxes,))
            misclassification_array = torch.zeros((num_gt_boxes,))

            # Only consider the model's confident predictions
            assert y_pred["scores"] is not None
            conf_pred_indices = torch.where(y_pred["scores"] > self.score_threshold)[0]
            if self.class_list:
                # Filter out predictions from classes not in class_list kwarg
                conf_pred_indices = conf_pred_indices[
                    torch.isin(y_pred["labels"][conf_pred_indices], self.class_list)
                ]

            # For each confident prediction
            for y_pred_idx in conf_pred_indices:
                y_pred_box = y_pred["boxes"][y_pred_idx]

                # Compute the iou between the predicted box and the ground-truth boxes
                ious = (
                    intersection_over_union(
                        torch.stack([y_pred_box]), gt_boxes, aggregate=False
                    )[0]
                    if len(gt_boxes) > 0
                    else torch.tensor([])
                )

                # Determine which ground-truth boxes, if any, the predicted box overlaps with
                overlap_indices = torch.where(ious > self.iou_threshold)[0]

                # If the predicted box doesn't overlap with any ground-truth boxes, increment
                # the hallucination counter and move on to the next predicted box
                if len(overlap_indices) == 0:
                    num_hallucinations += 1
                    continue

                # For each ground-truth box that the prediction overlaps with
                for y_idx in overlap_indices:
                    # If the predicted label is correct, mark that the ground-truth
                    # box has a true positive prediction
                    if gt_labels[y_idx] == y_pred["labels"][y_pred_idx]:
                        true_positive_array[y_idx] = 1
                    else:
                        # Otherwise mark that the ground-truth box has a misclassification
                        misclassification_array[y_idx] = 1

            true_positive_rate = (
                true_positive_array.mean()
                if len(true_positive_array) > 0
                else torch.tensor(0)
            )
            misclassification_rate = (
                misclassification_array.mean()
                if len(misclassification_array) > 0
                else torch.tensor(0)
            )

            # Any ground-truth box that had no overlapping predicted box is considered a
            # disappearance
            disappearance_rate = 1 - true_positive_rate - misclassification_rate

            self.true_positive_rate_per_img.append(true_positive_rate)
            self.misclassification_rate_per_img.append(misclassification_rate)
            self.disappearance_rate_per_img.append(disappearance_rate)
            self.hallucinations_per_img.append(num_hallucinations)

    def compute(self):
        true_positive_rate_per_img = dim_zero_cat(self.true_positive_rate_per_img)
        misclassification_rate_per_img = dim_zero_cat(
            self.misclassification_rate_per_img
        )
        disappearance_rate_per_img = dim_zero_cat(self.disappearance_rate_per_img)
        hallucinations_per_img = dim_zero_cat(self.hallucinations_per_img).to(
            dtype=torch.float32
        )

        if self.mean:
            return {
                "true_positive_rate_mean": torch.mean(true_positive_rate_per_img),
                "misclassification_rate_mean": torch.mean(
                    misclassification_rate_per_img
                ),
                "disappearance_rate_mean": torch.mean(disappearance_rate_per_img),
                "hallucinations_mean": torch.mean(hallucinations_per_img),
            }

        return {
            "true_positive_rate_per_img": true_positive_rate_per_img,
            "misclassification_rate_per_img": misclassification_rate_per_img,
            "disappearance_rate_per_img": disappearance_rate_per_img,
            "hallucinations_per_img": hallucinations_per_img,
        }
