from functools import partial
from typing import Optional

import torch

from armory.data import (
    BBoxFormat,
    BoundingBoxSpec,
    DataType,
    ImageDimensions,
    ImageSpec,
    ObjectDetectionBatch,
    Scale,
    TorchBoundingBoxSpec,
    TorchImageSpec,
)
from armory.model.object_detection.object_detector import ObjectDetector
from armory.track import track_init_params


@track_init_params
class YoloV5ObjectDetector(ObjectDetector):
    """
    Model wrapper with pre-applied output adapters for Ultralytics YOLOv5 models.

    Example::

        import yolov5
        from armory.model.object_detection import Yolov5ObjectDetector

        model = yolov5.load_model(CHECKPOINT)

        detector = YoloV5ObjectDetector(
            name="My model",
            model=model,
        )
    """

    def __init__(
        self,
        name: str,
        model,
        inputs_spec: Optional[ImageSpec] = None,
        predictions_spec: Optional[BoundingBoxSpec] = None,
        **kwargs,
    ):
        """
        Initializes the model wrapper.

        Args:
            name: Name of the model.
            model: YOLOv5 model being wrapped.
            inputs_spec: Optional, data specification used to obtain raw image
                data from the image inputs contained in object detection
                batches. Defaults to a specification compatible with typical
                YOLOv5 models.
            predictions_spec: Optional, data specification used to update the
                object detection predictions in the batch. Defaults to a
                bounding box specification compatible with typical YOLOv5 models.
            **kwargs: All other keyword arguments will be forwarded to the
                `yolov5.utils.general.non_max_suppression` function used to
                postprocess the model outputs.
        """
        super().__init__(
            name=name,
            model=model,
            inputs_spec=(
                inputs_spec
                or TorchImageSpec(
                    dim=ImageDimensions.CHW,
                    scale=Scale(dtype=DataType.FLOAT, max=1.0),
                    dtype=torch.float32,
                )
            ),
            predictions_spec=(
                predictions_spec or TorchBoundingBoxSpec(format=BBoxFormat.XYXY)
            ),
        )

        from yolov5.utils.general import non_max_suppression
        from yolov5.utils.loss import ComputeLoss

        self.compute_loss = ComputeLoss(self._model.model.model)
        self.nms = partial(non_max_suppression, **kwargs)

    def forward(self, x, targets=None):
        """
        Invokes the wrapped model. If in training and given targets, then the
        loss is computed and returned rather than the raw predictions.
        """
        # inputs: CHW images, 0.0-1.0 float
        # outputs: (N,6) detections (cx,cy,w,h,scores,labels)
        if self.training and targets is not None:
            outputs = self._model.model.model(x)
            loss, _ = self.compute_loss(outputs, targets)
            return dict(loss_total=loss)
        preds = self._model(x)
        return preds

    def predict(self, batch: ObjectDetectionBatch):
        """
        Invokes the wrapped model using the image inputs in the given batch and
        updates the object detection predictions in the batch.

        Non-maximum suppression processing is applied to the model's outputs
        before the batch predictions are updated.

        Args:
            batch: Object detection batch
        """
        self.eval()
        inputs = batch.inputs.get(self.inputs_spec)
        outputs = self(inputs)
        outputs = self.nms(outputs)
        outputs = [
            {
                "boxes": output[:, 0:4],
                "labels": torch.argmax(output[:, 5:], dim=1, keepdim=False),
                "scores": output[:, 4],
            }
            for output in outputs
        ]
        batch.predictions.set(outputs, self.predictions_spec)
