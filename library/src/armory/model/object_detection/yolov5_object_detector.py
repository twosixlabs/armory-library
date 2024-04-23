from functools import partial
from typing import Optional

import torch
from yolov5.models.yolo import DetectionModel

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
        model: DetectionModel,
        inputs_accessor: Optional[Images.Accessor] = None,
        predictions_accessor: Optional[Accessor] = None,
        iou_threshold: Optional[float] = None,
        score_threshold: Optional[float] = None,
        **kwargs,
    ):
        """
        Initializes the model wrapper.

        Args:
            name: Name of the model.
            model: YOLOv5 model being wrapped.
            inputs_accessor: Optional, data accessor used to obtain low-level
                image data from the highly-structured image inputs contained in
                object detection batches. Defaults to an accessor compatible
                with typical YOLOv5 models.
            predictions_accessor: Optional, data accessor used to update the
                object detection predictions in the batch. Defaults to an
                accessor compatible with typical YOLOv5 models.
            **kwargs: All other keyword arguments will be forwarded to the
                `yolov5.utils.general.non_max_suppression` function used to
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
                predictions_accessor or BoundingBoxes.as_torch(format=BBoxFormat.XYXY)
            ),
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )

        from yolov5.utils.general import non_max_suppression
        from yolov5.utils.loss import ComputeLoss

        self._detection_model = self._get_detection_model(self._model)
        self.compute_loss = ComputeLoss(self._detection_model)

        kwargs.update(
            {
                **({"conf_thres": score_threshold} if score_threshold else {}),
                **({"iou_thres": iou_threshold} if iou_threshold else {}),
            }
        )
        self.nms = partial(non_max_suppression, **kwargs)

    @staticmethod
    def _get_detection_model(model: DetectionModel) -> DetectionModel:
        detect_model = model
        try:
            while type(detect_model) is not DetectionModel:
                detect_model = detect_model.model
        except AttributeError:
            raise TypeError(f"{model} is not a {DetectionModel.__name__}")
        return detect_model

    def forward(self, x, targets=None):
        """
        Invokes the wrapped model. If in training and given targets, then the
        loss is computed and returned rather than the raw predictions. This is
        required when YoloV5ObjectDetector is used with ART and is wrapped by
        art.estimators.object_detection.PyTorchYolo.
        """
        # inputs: CHW images, 0.0-1.0 float
        # outputs: (N,6) detections (cx,cy,w,h,scores,labels)
        if self.training and targets is not None:
            outputs = self._detection_model(x)
            loss, _ = self.compute_loss(outputs, targets)
            return dict(loss_total=loss)
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
        self.eval()
        inputs = self.inputs_accessor.get(batch.inputs)
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
        self.predictions_accessor.set(batch.predictions, outputs)
