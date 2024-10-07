import logging
from typing import Optional

import numpy as np
import torchvision.ops

from armory.data import (
    BoundingBoxSpec,
    ImageSpec,
    ObjectDetectionBatch,
    TorchSpec,
    to_numpy,
    to_torch,
)
from armory.evaluation import ModelProtocol
from armory.model.base import ArmoryModel, ModelInputAdapter, ModelOutputAdapter

_logger: logging.Logger = logging.getLogger(__name__)


class ObjectDetector(ArmoryModel, ModelProtocol):
    """
    Wrapper around a model that produces object detection predictions.

    This wrapper applies score-threshold filtering and non-maximum suppression
    to the model's outputs if configured with a score threshold or an IOU
    threshold.

    Example::

        import armory.data
        from armory.model.object_detection import ObjectDetector

        # assuming `model` has been defined elsewhere
        detector = ObjectDetector(
            name="My model",
            model=model,
            inputs_spec=armory.data.TorchImageSpec(
                dim=armory.data.ImageDimensions.CHW,
                scale=armory.data.Scale(
                    dtype=armory.data.DataType.FLOAT,
                    max=1.0,
                ),
            ),
            predictions_spec=armory.data.BoundingBoxSpec(
                format=armory.data.BBoxFormat.XYXY
            ),
        )
    """

    def __init__(
        self,
        name: str,
        model,
        inputs_spec: ImageSpec,
        predictions_spec: BoundingBoxSpec,
        preadapter: Optional[ModelInputAdapter] = None,
        postadapter: Optional[ModelOutputAdapter] = None,
        iou_threshold: Optional[float] = None,
        score_threshold: Optional[float] = None,
    ):
        """
        _summary_

        :param name: Name of the model
        :type name: str
        :param model: Object detection model being wrapped
        :type model: _type_
        :param inputs_spec: Data specification used to obtain raw image data from
                the image inputs contained in object detection batches.
        :type inputs_spec: ImageSpec
        :param predictions_spec: Data specification used to update the raw object
                detection predictions data in the batch.
        :type predictions_spec: BoundingBoxSpec
        :param preadapter: Optional, model input adapter., defaults to None
        :type preadapter: ModelInputAdapter, optional
        :param postadapter: Optional, model output adapter., defaults to None
        :type postadapter: ModelOutputAdapter, optional
        :param iou_threshold: Optional, IOU threshold for non-maximum suppression, defaults to None
        :type iou_threshold: float, optional
        :param score_threshold: Optional, minimum score for predictions to be
                retained, defaults to None
        :type score_threshold: float, optional
        """
        super().__init__(
            name=name,
            model=model,
            preadapter=preadapter,
            postadapter=postadapter,
        )
        self.inputs_spec = inputs_spec
        self.predictions_spec = predictions_spec
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

    def _apply(self, *args, **kwargs):
        res = super()._apply(*args, **kwargs)
        if isinstance(self.inputs_spec, TorchSpec):
            self.inputs_spec.to(device=self.device)
        return res

    def _filter_predictions(self, preds):
        for pred in preds:
            # Filter based on score shreshold, if configured
            if self.score_threshold is not None:
                _logger.debug(
                    f"started with {len(pred['boxes'])} boxes before score threshold"
                )
                keep = pred["scores"] > self.score_threshold
                pred["boxes"] = pred["boxes"][keep]
                pred["scores"] = pred["scores"][keep]
                pred["labels"] = pred["labels"][keep]
                _logger.debug(f"keeping {len(pred['boxes'])} boxes")
            # Perform non-maximum suppression, if configured
            if self.iou_threshold is not None:
                _logger.debug(
                    f"started with {len(pred['boxes'])} boxes before iou threshold"
                )
                keep = torchvision.ops.nms(
                    boxes=to_torch(pred["boxes"]),
                    scores=to_torch(pred["scores"]),
                    iou_threshold=self.iou_threshold,
                )

                if isinstance(pred["scores"], np.ndarray):
                    keep = to_numpy(keep)

                pred["boxes"] = pred["boxes"][keep]
                pred["scores"] = pred["scores"][keep]
                pred["labels"] = pred["labels"][keep]
                _logger.debug(f"keeping {len(pred['boxes'])} boxes")
        return preds

    def predict(self, batch: ObjectDetectionBatch):
        """
        Invokes the wrapped model using the image inputs in the given batch and
        updates the object detection predictions in the batch.

        If the wrapper has been configured with a score threshold or an IOU
        threshold, the predictions will be filtered before the batch predictions
        are updated.

        :param batch: Object detection batch
        :type batch: ObjectDetectionBatch
        """
        self.eval()
        inputs = batch.inputs.get(self.inputs_spec)
        outputs = self(inputs)
        outputs = self._filter_predictions(outputs)
        batch.predictions.set(outputs, self.predictions_spec)
