"""Object detection evaluation task"""

import torch
import torchmetrics.detection

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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.benign_map = MAP(prefix="benign", class_metrics=class_metrics)
        self.attack_map = MAP(prefix="attack", class_metrics=class_metrics)
        self.export_score_threshold = export_score_threshold

    def export_batch(self, batch: BaseEvaluationTask.Batch):
        self._export("benign", batch.x, batch.y, batch.y_pred, batch.i)
        if batch.x_adv is not None:
            self._export("attack", batch.x_adv, batch.y, batch.y_pred_adv, batch.i)

    def _export(self, name, images, truth, preds, batch_idx):
        batch_size = images.shape[0]
        for sample_idx in range(batch_size):
            boxes_above_threshold = preds[sample_idx]["boxes"][
                preds[sample_idx]["scores"] > self.export_score_threshold
            ]
            with_boxes = draw_boxes_on_image(
                image=images[sample_idx],
                ground_truth_boxes=truth[sample_idx]["boxes"],
                pred_boxes=boxes_above_threshold,
            )
            filename = f"batch_{batch_idx}_ex_{sample_idx}_{name}.png"
            self.exporter.log_image(with_boxes, filename)

    def run_benign(self, batch: BaseEvaluationTask.Batch):
        super().run_benign(batch)
        if batch.y_pred is not None:
            self.benign_map.update(batch.y_pred, batch.y)

    def run_attack(self, batch: BaseEvaluationTask.Batch):
        super().run_attack(batch)
        if batch.y_pred_adv is not None:
            self.attack_map.update(batch.y_pred_adv, batch.y)

    def on_test_epoch_end(self):
        if not self.skip_benign:
            self.log_dict(self.benign_map.compute(), sync_dist=True)
            self.benign_map.reset()
        if not self.skip_attack:
            self.log_dict(self.attack_map.compute(), sync_dist=True)
            self.attack_map.reset()
        return super().on_validation_epoch_end()
