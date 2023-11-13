"""Object detection evaluation task"""

from typing import Optional

import numpy as np
import torch
import torchmetrics.detection
import torchvision.ops

from charmory.export import draw_boxes_on_image
from charmory.tasks.base import BaseEvaluationTask


class MAP(torchmetrics.detection.MeanAveragePrecision):
    """Extension of torchmetrics MeanAveragePrecision to make it work with Armory & Lightning"""

    def __init__(self, prefix: str, **kwargs):
        super().__init__(**kwargs)
        self.prefix = prefix

    def update(self, preds, target):
        """
        Convert predictions and target from Armory/ART types to torchmetrics-compatible types
        """
        super().update(self._create_map_preds(preds), self._create_map_target(target))

    @staticmethod
    def _create_map_preds(preds):
        return [
            {
                "boxes": torch.FloatTensor(pred["boxes"]),
                "scores": torch.FloatTensor(pred["scores"]),
                "labels": torch.IntTensor(pred["labels"]),
            }
            for pred in preds
        ]

    @staticmethod
    def _create_map_target(targets):
        return [
            {
                "boxes": torch.FloatTensor(target["boxes"]),
                "labels": torch.IntTensor(target["labels"]),
            }
            for target in targets
        ]

    def compute(self) -> dict:
        """
        Convert torchmetrics results to Lightning-compatible metrics
        """
        metrics = {f"{self.prefix}_{k}": v for k, v in super().compute().items()}
        metrics.pop(f"{self.prefix}_classes")
        maps_per_class = metrics.pop(f"{self.prefix}_map_per_class")
        if self.class_metrics and maps_per_class is not None:
            metrics.update(
                {
                    f"{self.prefix}_map_class_{id}": value
                    for id, value in enumerate(maps_per_class)
                    if value != -1
                }
            )
        mars_per_class = metrics.pop(f"{self.prefix}_mar_100_per_class")
        if self.class_metrics and mars_per_class is not None:
            metrics.update(
                {
                    f"{self.prefix}_mar_class_{id}": value
                    for id, value in enumerate(mars_per_class)
                    if value != -1
                }
            )
        return metrics


class ObjectDetectionTask(BaseEvaluationTask):
    """Object detection evaluation task"""

    def __init__(
        self,
        *args,
        class_metrics: bool = False,
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
            class_metrics: Whether to track mAP metrics per class
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
        self.benign_map = MAP(prefix="benign", class_metrics=class_metrics)
        self.attack_map = MAP(prefix="attack", class_metrics=class_metrics)
        self.export_score_threshold = export_score_threshold
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

    def export_batch(self, batch: BaseEvaluationTask.Batch):
        self._export_image("benign", batch.x, batch.y, batch.y_pred, batch.i)
        if batch.x_adv is not None:
            self._export_image(
                "attack", batch.x_adv, batch.y, batch.y_pred_adv, batch.i
            )
        self._export_targets(batch)

    def _export_image(self, name, images, truth, preds, batch_idx):
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
            filename = f"batch_{batch_idx}_ex_{sample_idx}_{name}.png"
            self.exporter.log_image(with_boxes, filename)

    @staticmethod
    def _from_list(maybe_list, idx):
        try:
            return maybe_list[idx]
        except:  # noqa: E722
            # if it's None or is not a list/sequence/etc, just return None
            return None

    def _export_targets(self, batch: BaseEvaluationTask.Batch):
        keys = set(batch.data.keys()) - {
            self.evaluation.dataset.x_key,
            self.evaluation.dataset.y_key,
        }
        for sample_idx in range(batch.x.shape[0]):
            dictionary = dict(
                y=batch.y[sample_idx],
                y_pred=self._from_list(batch.y_pred, sample_idx),
                y_target=self._from_list(batch.y_target, sample_idx),
                y_pred_adv=self._from_list(batch.y_pred_adv, sample_idx),
            )
            for k in keys:
                dictionary[k] = self._from_list(batch.data[k], sample_idx)
            self.exporter.log_dict(
                dictionary=dictionary,
                artifact_file=f"batch_{batch.i}_ex_{sample_idx}_y.txt",
            )

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

    def run_benign(self, batch: BaseEvaluationTask.Batch):
        super().run_benign(batch)
        if batch.y_pred is not None:
            batch.y_pred = self._filter_predictions(batch.y_pred)
            self.benign_map.update(batch.y_pred, batch.y)

    def run_attack(self, batch: BaseEvaluationTask.Batch):
        super().run_attack(batch)
        if batch.y_pred_adv is not None:
            batch.y_pred_adv = self._filter_predictions(batch.y_pred_adv)
            self.attack_map.update(batch.y_pred_adv, batch.y)

    def on_test_epoch_end(self):
        if not self.skip_benign:
            self.log_dict(self.benign_map.compute(), sync_dist=True)
            self.benign_map.reset()
        if not self.skip_attack:
            self.log_dict(self.attack_map.compute(), sync_dist=True)
            self.attack_map.reset()
        return super().on_validation_epoch_end()
