"""Armory model wrapper for HuggingFace transformer YOLOS models."""
from typing import TYPE_CHECKING, Optional, Tuple

import torch

if TYPE_CHECKING:
    from transformers.models.yolos import YolosImageProcessor, YolosModel

from armory.data import (
    Accessor,
    BBoxFormat,
    BoundingBoxes,
    DataType,
    ImageDimensions,
    Images,
    Scale,
)
from armory.model.object_detection.object_detector import ObjectDetector


class YolosTransformer(ObjectDetector):
    """
    Model wrapper with pre-applied input and output adapters for HuggingFace
    transformer YOLOS models.

    Example::

        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        from armory.model.object_detection import YolosTransformer

        model = AutoModelForObjectDetection.from_pretrained(CHECKPOINT)
        processor = AutoImageProcessor.from_pretrained(CHECKPOINT)

        transformer = YolosTransformer(
            name="My model",
            model=model,
            image_processor=processor,
        )
    """

    DEFAULT_MEAN = (0.485, 0.456, 0.406)
    DEFAULT_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        name: str,
        model: "YolosModel",
        image_processor: "YolosImageProcessor",
        inputs_accessor: Optional[Images.Accessor] = None,
        predictions_accessor: Optional[Accessor] = None,
        target_size: Tuple[int, int] = (512, 512),
    ):
        """
        Initializes the model wrapper.

        Args:
            name: Name of the model.
            model: YOLOS model being wrapped.
            image_processor: HuggingFace YOLOS image processor corresponding to
                the model.
            inputs_accessor: Optional, data accessor used to obtain low-level
                image data from the highly-structured image inputs contained in
                object detection batches. Defaults to an accessor compatible
                with typical YOLOS models.
            predictions_accessor: Optional, data accessor used to update the
                object detection predictions in the batch. Defaults to an
                accessor compatible with typical YOLOS models.
            target_size: Size (as a `height, width` tuple) of images, used for
                correct postprocessing and resizing of the bounding box
                predictions.
        """
        super().__init__(
            name=name,
            model=model,
            preadapter=self._preadapt,
            postadapter=self._postadapt,
            inputs_accessor=(
                inputs_accessor
                or Images.as_torch(
                    dim=ImageDimensions.CHW,
                    scale=Scale(
                        dtype=DataType.FLOAT,
                        max=1.0,
                        mean=self.DEFAULT_MEAN,
                        std=self.DEFAULT_STD,
                    ),
                    dtype=torch.float32,
                )
            ),
            predictions_accessor=(
                predictions_accessor or BoundingBoxes.as_torch(format=BBoxFormat.XYXY)
            ),
        )
        self.image_processor = image_processor
        self.target_size = target_size

    def _preadapt(self, images, targets=None, **kwargs):
        # Prediction targets need a `class_labels` property rather than the
        # `labels` property that's being passed in from ART
        if targets is not None:
            for target in targets:
                target["class_labels"] = target["labels"]

        return (images, targets), kwargs

    def _postadapt(self, output):
        # The model is put in training mode during attack generation, and
        # we need to return the loss components instead of the predictions
        if output.loss_dict is not None:
            return output.loss_dict

        result = self.image_processor.post_process_object_detection(
            output,
            target_sizes=[self.target_size for _ in range(len(output.pred_boxes))],
        )
        return result
