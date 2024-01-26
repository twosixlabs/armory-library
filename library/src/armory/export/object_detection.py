from typing import Optional

import numpy as np
import torch
from torchvision.utils import draw_bounding_boxes

from armory.data import (
    Batch,
    BBoxFormat,
    BoundingBoxes,
    DataType,
    ImageDimensions,
    Images,
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
        score_threshold: float = 0.5,
        inputs_accessor: Optional[Images.Accessor] = None,
        predictions_accessor: Optional[BoundingBoxes.Accessor] = None,
        targets_accessor: Optional[BoundingBoxes.Accessor] = None,
    ):
        """
        Initializes the exporter.

        Args:
            score_threshold: Optional, minimum score for object detection
                predictions to be included as drawn bounding boxes in the
                exported images. Defaults to 0.5.
            inputs_accessor: Optional, data exporter used to obtain low-level
                image data from the highly-structured inputs contained in
                exported batches. By default, a NumPy images accessor is used.
            predictions_accessor: Optional, data exporter used to obtain
                low-level predictions data from the highly-structured
                predictions contained in exported batches. By default, an XYXY
                NumPy bounding box accessor is used.
            targets_accessor: Optional, data exporter used to obtain low-level
                ground truth targets data from the high-ly structured targets
                contained in exported batches. By default, an XYXY NumPy
                bounding box accessor is used.
        """
        super().__init__(
            predictions_accessor=(
                predictions_accessor or BoundingBoxes.as_numpy(format=BBoxFormat.XYXY)
            ),
            targets_accessor=(
                targets_accessor or BoundingBoxes.as_numpy(format=BBoxFormat.XYXY)
            ),
        )
        self.score_threshold = score_threshold
        self.inputs_accessor = inputs_accessor or Images.as_numpy(
            dim=ImageDimensions.CHW,
            scale=Scale(dtype=DataType.UINT8, max=255),
            dtype=np.uint8,
        )

    def export(self, chain_name: str, batch_idx: int, batch: Batch) -> None:
        assert self.sink, "No sink has been set, unable to export"

        self._export_metadata(chain_name, batch_idx, batch)

        images = self.inputs_accessor.get(batch.inputs)
        targets = self.targets_accessor.get(batch.targets)
        predictions = self.predictions_accessor.get(batch.predictions)
        for sample_idx, image in enumerate(images):
            boxes_above_threshold = predictions[sample_idx]["boxes"][
                predictions[sample_idx]["scores"] > self.score_threshold
            ]
            # We access images as CHW because draw_bounding_boxes requires it,
            # but MLFlow needs HWC so we transpose
            with_boxes = draw_boxes_on_image(
                image=image,
                ground_truth_boxes=targets[sample_idx]["boxes"],
                pred_boxes=boxes_above_threshold,
            ).transpose(1, 2, 0)
            filename = f"batch_{batch_idx}_ex_{sample_idx}_{chain_name}.png"
            self.sink.log_image(with_boxes, filename)
