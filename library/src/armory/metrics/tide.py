"""TIDE metrics"""

from typing import Iterable, Optional

import tidecv
import tidecv.data
from torchmetrics import Metric

from armory.data import BBoxFormat, TorchBoundingBoxSpec
from armory.metric import PredictionMetric


class TIDE(Metric):
    """
    A torchmetrics-compatible interface for TIDE (https://github.com/dbolya/tide)
    detection metrics
    """

    @classmethod
    def create(
        cls,
        record_as_artifact: bool = True,
        record_as_metrics: Optional[Iterable[str]] = None,
    ):
        """
        Creates an instance of the TIDE metric pre-wrapped in a
        `armory.metric.PredictionMetric`

        Args:
            record_as_artifact: If True, the metric result will be recorded as
                an artifact to the evaluation run.
            record_as_metrics: Optional, a set of JSON paths in the metric
                result pointing to scalar values to record as metrics to the
                evaluation run. If None, no metrics will be recorded.

        Return:
            Armory prediction metric for TIDE detection metrics
        """
        return PredictionMetric(
            metric=cls(),
            spec=TorchBoundingBoxSpec(format=BBoxFormat.XYWH),
            record_as_artifact=record_as_artifact,
            record_as_metrics=record_as_metrics,
        )

    def __init__(self):
        super().__init__()
        self.add_state("preds_boxes", default=[], dist_reduce_fx=None)
        self.add_state("preds_labels", default=[], dist_reduce_fx=None)
        self.add_state("preds_scores", default=[], dist_reduce_fx=None)
        self.add_state("target_boxes", default=[], dist_reduce_fx=None)
        self.add_state("target_labels", default=[], dist_reduce_fx=None)

    def update(self, preds, target):
        self.preds_boxes.extend([p["boxes"] for p in preds])
        self.preds_labels.extend([p["labels"] for p in preds])
        self.preds_scores.extend([p["scores"] for p in preds])
        self.target_boxes.extend([t["boxes"] for t in target])
        self.target_labels.extend([t["labels"] for t in target])

    def compute(self):
        data_preds = tidecv.data.Data(name="preds")
        data_target = tidecv.data.Data(name="target")

        max_dets = 0
        image_id = 0
        for preds_boxes, preds_labels, preds_scores, target_boxes, target_labels in zip(
            self.preds_boxes,
            self.preds_labels,
            self.preds_scores,
            self.target_boxes,
            self.target_labels,
        ):
            image_id += 1
            for box, label in zip(target_boxes, target_labels):
                data_target.add_ground_truth(
                    image_id=image_id,
                    class_id=label.item(),
                    box=box.cpu().numpy(),
                )

            for box, score, label in zip(preds_boxes, preds_scores, preds_labels):
                data_preds.add_detection(
                    image_id=image_id,
                    class_id=label.item(),
                    score=score.item(),
                    box=box.cpu().numpy(),
                )

            max_dets = max(max_dets, max(len(preds_labels), len(target_labels)))

        data_preds.max_dets = max_dets
        data_target.max_dets = max_dets

        tide = tidecv.TIDE()
        tide.evaluate_range(data_target, data_preds, mode=tidecv.TIDE.BOX)

        errors = tide.get_all_errors()

        return {
            "mAP": {x.pos_thresh: x.ap / 100 for x in tide.run_thresholds["preds"]},
            "errors": {
                "main": {
                    "dAP": {k: v / 100 for k, v in errors["main"]["preds"].items()},
                    "count": {
                        k.short_name: len(v)
                        for k, v in tide.runs["preds"].error_dict.items()
                    },
                },
                "special": {
                    "dAP": {k: v / 100 for k, v in errors["special"]["preds"].items()},
                    "count": {
                        "FalseNeg": sum(
                            map(len, tide.runs["preds"].false_negatives.values())
                        )
                    },
                },
            },
        }
