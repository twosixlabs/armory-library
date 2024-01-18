from functools import partial
from typing import Optional

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


@track_init_params
class YoloV5ObjectDetector(ObjectDetector):
    def __init__(
        self,
        name: str,
        model,
        inputs_accessor: Optional[Images.Accessor] = None,
        predictions_accessor: Optional[Accessor] = None,
        **kwargs,
    ):
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
        )

        from yolov5.utils.general import non_max_suppression
        from yolov5.utils.loss import ComputeLoss

        self.compute_loss = ComputeLoss(self._model.model.model)
        self.nms = partial(non_max_suppression, **kwargs)

    def forward(self, x, targets=None):
        # inputs: CHW images, 0.0-1.0 float
        # outputs: (N,6) detections (cx,cy,w,h,scores,labels)
        if self.training and targets is not None:
            outputs = self._model.model.model(x)
            loss, _ = self.compute_loss(outputs, targets)
            return dict(loss_total=loss)
        preds = self._model(x)
        return preds

    def predict(self, batch: Batch):
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
