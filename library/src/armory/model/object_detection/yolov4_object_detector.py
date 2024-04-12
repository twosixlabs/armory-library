from functools import partial
from typing import Optional
import numpy as np
import sys
import torch

from armory.data import (
    Accessor,
    Batch,
    BBoxFormat,
    BoundingBoxes,
    DataType,
    ImageDimensions,
    Images,
    Scale,
)
from armory.model.object_detection.object_detector import ObjectDetector
from armory.track import track_init_params
from armory.model.object_detection.yolov4_detect_utils import post_processing


@track_init_params
class YoloV4ObjectDetector(ObjectDetector):
    """
    Model wrapper with pre-applied output adapters for Ultralytics YOLOv4 models.

    Example::

        import yolov4
        from armory.model.object_detection import Yolov4ObjectDetector

        model = yolov4.load_model(CHECKPOINT)

        detector = YoloV4ObjectDetector(
            name="My model",
            model=model,
        )
    """

    def __init__(
        self,
        name: str,
        model,
        inputs_accessor: Optional[Images.Accessor] = None,
        predictions_accessor: Optional[Accessor] = None,
        **kwargs,
    ):
        """
        Initializes the model wrapper.

        Args:
            name: Name of the model.
            model: YOLOv4 model being wrapped.
            inputs_accessor: Optional, data accessor used to obtain low-level
                image data from the highly-structured image inputs contained in
                object detection batches. Defaults to an accessor compatible
                with typical YOLOv4 models.
            predictions_accessor: Optional, data accessor used to update the
                object detection predictions in the batch. Defaults to an
                accessor compatible with typical YOLOv4 models.
            **kwargs: All other keyword arguments will be forwarded to the
                `yolov4_detect_utils.general.non_max_suppression` function used to
                postprocess the model outputs.
        """
        super().__init__(
            name=name,
            model=model,
            inputs_accessor=(
                inputs_accessor
                or Images.as_torch(
                    dim=ImageDimensions.CHW,
                    scale=Scale(dtype=DataType.FLOAT, max=1.0),
                    dtype=torch.float32,
                )
            ),
            predictions_accessor=(
                predictions_accessor or BoundingBoxes.as_torch(
                    format=BBoxFormat.XYXY)
            ),
        )
        # TODO: From YoloV5ObjectDetector, rework for YoloV4ObjectDetector
        # self.compute_loss = ComputeLoss(self._model.model.model)
        # self.nms = partial(non_max_suppression, **kwargs)

    def forward(self, x, targets=None):
        """
        Invokes the wrapped model. If in training and given targets, then the
        loss is computed and returned rather than the raw predictions.
        """
        # TODO: From YoloV5ObjectDetector, rework this for YoloV4ObjectDetector
        # inputs: CHW images, 0.0-1.0 float
        # outputs: (N,6) detections (cx,cy,w,h,scores,labels)
        # if self.training and targets is not None:
        #     outputs = self._model.model.model(x)
        #     loss, _ = self.compute_loss(outputs, targets)
        #     return dict(loss_total=loss)
        preds = self._model(x)
        return preds

    def predict(self, batch: Batch):
        """
        Invokes the wrapped model using the image inputs in the given batch and
        updates the object detection predictions in the batch.

        Non-maximum suppression processing is applied to the model's outputs
        before the batch predictions are updated.

        Args:
            batch: Object detection batch
        """
        # Temporarily hardcoded wp_yolov4 image dimensions
        w = 608
        h = 608

        self.eval()
        inputs = self.inputs_accessor.get(batch.inputs)
        outputs = self(inputs)
        outputs = post_processing(outputs)

        if all(not sublist for sublist in outputs):
            outputs_dict = [
                {
                    "boxes": np.array([]),
                    "labels": np.array([]),
                    "scores": np.array([]),
                }
                for output in outputs
            ]
            self.predictions_accessor.set(batch.predictions, outputs_dict)
        else:
            outputs_dict = [
                {
                    "boxes": np.array(
                        [[output[0]*w, output[1]*h, output[2]*w, output[3]*h] for output in outputs[0][:4]], dtype=np.float32
                    ),
                    "labels": (
                        labels := np.array(
                            [int(output[5]) for output in outputs[0]], dtype=np.int64
                        )
                    ),
                    "scores": np.array(
                        [float(output[4]) for output in outputs[0]], dtype=np.float32
                    )
                }
                for output in outputs
            ]
            self.predictions_accessor.set(batch.predictions, outputs_dict)
