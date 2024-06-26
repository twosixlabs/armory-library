from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Tuple

import torch

from armory.data import (
    BBoxFormat,
    BoundingBoxes,
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

if TYPE_CHECKING:
    from yolov5.models.yolo import DetectionModel


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
        detection_model: Optional["DetectionModel"] = None,
        inputs_spec: Optional[ImageSpec] = None,
        predictions_spec: Optional[BoundingBoxSpec] = None,
        targets_spec: Optional[BoundingBoxSpec] = None,
        iou_threshold: Optional[float] = None,
        score_threshold: Optional[float] = None,
        compute_loss: Optional[Callable[[Any, Any], Tuple[Any, Any]]] = None,
        **kwargs,
    ):
        """
        Initializes the model wrapper.

        Args:
            name: Name of the model.
            model: YOLOv5 model being wrapped.
            detection_model: Optional, the inner YOLOv5 detection model to use
                for computing loss. By default, the detection model is assumed
                to be a property of the inner model property of the given YOLOv5
                model--that is, `model.model.model`. It is unlikely that this
                argument will ever be necessary, and may only be required if
                the upstream `yolov5` package changes its model structure.
            inputs_spec: Optional, data specification used to obtain raw image
                data from the image inputs contained in object detection
                batches. Defaults to a specification compatible with typical
                YOLOv5 models.
            predictions_spec: Optional, data specification used to update the
                object detection predictions in the batch. Defaults to a
                bounding box specification compatible with typical YOLOv5 models.
            targets_spec: Optional, data specification used to obtain ground
                truth targets from object detection batches in order to
                calculate loss. Defaults to a bounding box specification
                compatible with typical YOLOv5 models.
            compute_loss: Optional, loss function used to calculate loss when the
                model is in training mode. By default, the standard YOLOv5 loss
                function is used. The function must accept two arguments: the
                model predictions and the ground truth targets. The function
                must return a tuple of the loss and the loss items (the second
                element is unused).
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
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )

        from yolov5.utils.general import non_max_suppression
        from yolov5.utils.loss import ComputeLoss

        self._detection_model: "DetectionModel" = (
            detection_model if detection_model is not None else self._model.model.model
        )
        self.compute_loss = (
            compute_loss
            if compute_loss is not None
            else ComputeLoss(self._detection_model)
        )
        self.targets_spec = targets_spec or TorchBoundingBoxSpec(
            format=BBoxFormat.CXCYWH
        )

        if score_threshold is not None:
            kwargs["conf_thres"] = score_threshold
        if iou_threshold is not None:
            kwargs["iou_thres"] = iou_threshold
        self.nms = partial(non_max_suppression, **kwargs)

    def forward(self, x, targets=None):
        """
        Invokes the wrapped model. If in training and given targets, then the
        loss is computed and returned rather than the raw predictions. This is
        required when YoloV5ObjectDetector is used with ART and is wrapped by
        art.estimators.object_detection.PyTorchYolo.
        """
        # inputs: CHW images, 0.0-1.0 float
        # outputs: (N,6) detections (cx,cy,w,h,scores,labels)
        if targets is not None:
            outputs = self._detection_model(x)
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
        outputs = self.nms(outputs)  # CXCYWH -> XYXY
        outputs = [
            {
                "boxes": output[:, 0:4],
                "labels": output[:, 5].to(torch.int64),
                "scores": output[:, 4],
            }
            for output in outputs
        ]
        batch.predictions.set(outputs, self.predictions_spec)

    def loss(self, batch: ObjectDetectionBatch) -> torch.Tensor:
        """
        Computes the loss for the given batch.

        Args:
            batch: Object detection batch

        Returns:
            The total loss
        """
        self.train()
        inputs = batch.inputs.get(self.inputs_spec)
        _, _, height, width = inputs.shape

        targets = batch.targets.get(self.targets_spec)
        yolo_targets = self._to_yolo_targets(targets, height, width, inputs.device)

        loss_components = self(inputs, yolo_targets)
        return loss_components["loss_total"]

    @staticmethod
    def _to_yolo_targets(
        targets: Sequence[BoundingBoxes.BoxesTorch],
        height: int,
        width: int,
        device,
    ) -> torch.Tensor:
        targets_list = []

        for i, target in enumerate(targets):
            labels = torch.zeros(len(target["boxes"]), 6, device=device)
            labels[:, 0] = i
            labels[:, 1] = target["labels"]
            labels[:, 2:6] = target["boxes"]

            # normalize bounding boxes to [0, 1]}
            labels[:, 2:6:2] = labels[:, 2:6:2] / width
            labels[:, 3:6:2] = labels[:, 3:6:2] / height

            targets_list.append(labels)

        r = torch.vstack(targets_list)
        return r
