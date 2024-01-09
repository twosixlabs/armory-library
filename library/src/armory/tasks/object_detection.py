"""Object detection evaluation task"""

from typing import Optional

import numpy as np
import torch
import torchvision.ops

from armory.export import draw_boxes_on_image
from armory.tasks.base import BaseEvaluationTask


class ObjectDetectionTask(BaseEvaluationTask):
    """Object detection evaluation task"""

    def __init__(
        self,
        *args,
        export_score_threshold: float = 0.5,
        iou_threshold: Optional[float] = None,
        score_threshold: Optional[float] = None,
        **kwargs,
    ):
        """
        Initializes the task.

        Args:
            *args: All positional arguments will be forwarded to the
                `charmory.tasks.base.BaseEvaluationTask` class
            export_score_threshold: Minimum prediction score for a detection
                bounding box to be drawn on the exported sample
            iou_threshold: Maximum intersection-over-union value for
                non-maximum suppression filtering of detection bounding boxes
            score_threshold: Minimum prediction score. All predictions with a
                score lower than this threshold will be ignored and not
                included in any evalutation metric
            **kwargs: All other keyword arguments will be forwarded to the
                `charmory.tasks.base.BaseEvaluationTask` class
        """
        super().__init__(*args, **kwargs)
        self.export_score_threshold = export_score_threshold
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

    def export_batch(self, batch: BaseEvaluationTask.Batch):
        if batch.x_perturbed is not None:
            self._export_image(
                batch.chain_name, batch.x_perturbed, batch.y, batch.y_predicted, batch.i
            )
        self.export_batch_metadata(batch)

    def _export_image(self, chain_name, images, truth, preds, batch_idx):
        batch_size = images.shape[0]
        for sample_idx in range(batch_size):
            image = images[sample_idx]
            if self.export_adapter is not None:
                image = self.export_adapter(image)
            boxes_above_threshold = preds[sample_idx]["boxes"][
                preds[sample_idx]["scores"] > self.export_score_threshold
            ]
            with_boxes = draw_boxes_on_image(
                image=image,
                ground_truth_boxes=truth[sample_idx]["boxes"],
                pred_boxes=boxes_above_threshold,
            )
            filename = f"batch_{batch_idx}_ex_{sample_idx}_{chain_name}.png"
            self.exporter.log_image(with_boxes, filename)

    def _filter_predictions(self, preds):
        for pred in preds:
            # Filter based on score shreshold, if configured
            if self.score_threshold is not None:
                keep = pred["scores"] > self.score_threshold
                pred["boxes"] = pred["boxes"][keep]
                pred["scores"] = pred["scores"][keep]
                pred["labels"] = pred["labels"][keep]
            # Perform non-maximum suppression, if configured
            if self.iou_threshold is not None:
                keep = torchvision.ops.nms(
                    boxes=torch.FloatTensor(pred["boxes"]),
                    scores=torch.FloatTensor(pred["scores"]),
                    iou_threshold=self.iou_threshold,
                )
                single = len(keep) == 1
                boxes = pred["boxes"][keep]
                scores = pred["scores"][keep]
                labels = pred["labels"][keep]
                pred["boxes"] = np.array([boxes]) if single else boxes
                pred["scores"] = np.array([scores]) if single else scores
                pred["labels"] = np.array([labels]) if single else labels
        return preds

    def evaluate(self, batch: BaseEvaluationTask.Batch):
        super().evaluate(batch)
        if batch.y_predicted is not None:
            batch.y_predicted = self._filter_predictions(batch.y_predicted)

    def target_to_tensor(self, targets):
        return [
            {
                "boxes": torch.FloatTensor(target["boxes"]),
                "labels": torch.IntTensor(target["labels"]),
                "scores": torch.FloatTensor(target.get("scores", [])),
            }
            for target in targets
        ]
