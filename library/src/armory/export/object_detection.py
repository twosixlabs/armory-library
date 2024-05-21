from typing import Iterable, Optional

import numpy as np
import torch
from torchvision.utils import draw_bounding_boxes

from armory.data import (
    BBoxFormat,
    DataType,
    ImageDimensions,
    NumpyBoundingBoxSpec,
    NumpyImageSpec,
    ObjectDetectionBatch,
    Scale,
)
from armory.export.base import Exporter


def draw_boxes_on_image(
    image: np.ndarray,
    ground_truth_boxes: Optional[np.ndarray] = None,
    ground_truth_color: str = "red",
    ground_truth_width: int = 2,
    pred_boxes: Optional[np.ndarray] = None,
    pred_color: str = "white",
    pred_width: int = 2,
) -> np.ndarray:
    """
    Draw bounding boxes for ground truth objects and predicted objects on top of
    an image sample.

    Ground truth bounding boxes will be drawn first, then the predicted bounding
    boxes.

    Args:
        image: 3-dimensional array of image data in shape (C, H, W) and type uint8
        ground_truth_boxes: Optional array of shape (N, 4) containing ground truth
            bounding boxes in (xmin, ymin, xmax, ymax) format.
        ground_truth_color: Color to use for ground truth bounding boxes. Color can
            be represented as PIL strings (e.g., "red").
        ground_truth_width: Width of ground truth bounding boxes.
        pred_boxes: Optional array of shape (N, 4) containing predicted
            bounding boxes in (xmin, ymin, xmax, ymax) format.
        pred_color: Color to use for predicted bounding boxes. Color can
            be represented as PIL strings (e.g., "red").
        pred_width: Width of ground truth bounding boxes.

    Return:
        uint8 tensor of (C, H, W) image with bounding boxes data
    """
    with_boxes = torch.as_tensor(image)

    if ground_truth_boxes is not None and len(ground_truth_boxes) > 0:
        with_boxes = draw_bounding_boxes(
            image=with_boxes,
            boxes=torch.as_tensor(ground_truth_boxes),
            colors=ground_truth_color,
            width=ground_truth_width,
        )

    if pred_boxes is not None and len(pred_boxes) > 0:
        with_boxes = draw_bounding_boxes(
            image=with_boxes,
            boxes=torch.as_tensor(pred_boxes),
            colors=pred_color,
            width=pred_width,
        )

    return with_boxes.numpy()


class ObjectDetectionExporter(Exporter):
    """An exporter for object detection samples."""

    def __init__(
        self,
        name: Optional[str] = None,
        score_threshold: float = 0.5,
        inputs_spec: Optional[NumpyImageSpec] = None,
        predictions_spec: Optional[NumpyBoundingBoxSpec] = None,
        targets_spec: Optional[NumpyBoundingBoxSpec] = None,
        criterion: Optional[Exporter.Criterion] = None,
    ):
        """
        Initializes the exporter.

        Args:
            name: Description of the exporter
            score_threshold: Optional, minimum score for object detection
                predictions to be included as drawn bounding boxes in the
                exported images. Defaults to 0.5.
            inputs_spec: Optional, data specification used to obtain raw
                image data from the inputs contained in exported batches. By
                default, a NumPy images specification is used.
            predictions_spec: Optional, data specification used to obtain raw
                predictions data from the exported batches. By default, an XYXY
                NumPy bounding box specification is used.
            targets_spec: Optional, data specification used to obtain raw ground
                truth targets data from the exported batches. By default, an XYXY
                NumPy bounding box specification is used.
            criterion: Criterion to determine when samples will be exported. If
                omitted, no samples will be exported.
        """
        super().__init__(
            name=name or "ObjectDetection",
            predictions_spec=(
                predictions_spec or NumpyBoundingBoxSpec(format=BBoxFormat.XYXY)
            ),
            targets_spec=(targets_spec or NumpyBoundingBoxSpec(format=BBoxFormat.XYXY)),
            criterion=criterion,
        )
        self.score_threshold = score_threshold
        self.inputs_spec = inputs_spec or NumpyImageSpec(
            dim=ImageDimensions.CHW,
            scale=Scale(dtype=DataType.UINT8, max=255),
            dtype=np.uint8,
        )
        self.targets_spec: NumpyBoundingBoxSpec
        self.predictions_spec: NumpyBoundingBoxSpec

    def export_samples(
        self, batch_idx: int, batch: ObjectDetectionBatch, samples: Iterable[int]
    ) -> None:
        assert self.sink, "No sink has been set, unable to export"

        self._export_metadata(batch_idx, batch, samples)

        images = batch.inputs.get(self.inputs_spec)
        targets = batch.targets.get(self.targets_spec)
        predictions = batch.predictions.get(self.predictions_spec)
        for sample_idx in samples:
            scores = predictions[sample_idx]["scores"]
            assert scores is not None
            boxes_above_threshold = predictions[sample_idx]["boxes"][
                scores > self.score_threshold
            ]
            # We access images as CHW because draw_bounding_boxes requires it,
            # but MLFlow needs HWC so we transpose
            with_boxes = draw_boxes_on_image(
                image=images[sample_idx],
                ground_truth_boxes=targets[sample_idx]["boxes"],
                pred_boxes=boxes_above_threshold,
            ).transpose(1, 2, 0)
            filename = self.artifact_path(batch_idx, sample_idx, "objects.png")
            self.sink.log_image(with_boxes, filename)
