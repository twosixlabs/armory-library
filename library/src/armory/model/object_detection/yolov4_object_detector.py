from typing import List, Optional, Tuple

import numpy as np
import torch

from armory.data import (
    BBoxFormat,
    BoundingBoxSpec,
    DataType,
    ImageDimensions,
    ImageSpec,
    NumpyBoundingBoxSpec,
    ObjectDetectionBatch,
    Scale,
    TorchImageSpec,
    to_numpy,
)
from armory.model.object_detection.object_detector import ObjectDetector
from armory.track import track_init_params


@track_init_params
class YoloV4ObjectDetector(ObjectDetector):
    """
    Model wrapper with pre-applied output adapters for YOLOv4 models.

    Example::

        from armory.model.object_detection import Yolov4ObjectDetector

        # assumes `model` has been created elsewhere
        detector = YoloV4ObjectDetector(
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
        iou_threshold: Optional[float] = None,
        score_threshold: Optional[float] = None,
    ):
        """
        Initializes the model wrapper.

        Args:
            name: Name of the model.
            model: YOLOv4 model being wrapped.
            inputs_spec: Optional, data specification used to obtain raw image
                data from the image inputs contained in object detection
                batches. Defaults to a specification compatible with typical
                YOLOv5 models.
            predictions_spec: Optional, data specification used to update the
                object detection predictions in the batch. Defaults to a
                bounding box specification compatible with typical YOLOv5 models.
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
                predictions_spec or NumpyBoundingBoxSpec(format=BBoxFormat.XYXY)
            ),
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )

    def predict(self, batch: ObjectDetectionBatch):
        """
        Invokes the wrapped model using the image inputs in the given batch and
        updates the object detection predictions in the batch.

        Non-maximum suppression processing is applied to the model's outputs
        before the batch predictions are updated.

        Args:
            batch: Object detection batch

        Model output / predictions format/example:
        outputs = self(inputs)
        Length of outputs: 2, type=list
        outputs[0] shape: torch.Size([1, 22743, 1, 4])
        outputs[1] shape: torch.Size([1, 22743, 3])

        predictions = _post_processing(outputs)[0]
        Length of predictions: 1, type=list[lists]
        predictions: [[-0.016737998, 0.051072836, 0.80007946, 0.96419203, 0.8180811, 0]]
        predictions[0]: [-0.016737998, 0.051072836, 0.80007946, 0.96419203, 0.8180811, 0]
        """
        self.eval()
        inputs = batch.inputs.get(self.inputs_spec)
        _, _, h, w = inputs.shape  # (N, C, H, W)
        outputs = self(inputs)
        predictions = _post_processing(outputs)[0]
        if all(not sublist for sublist in predictions):
            preds_dict = [
                {
                    "boxes": np.array([]),
                    "labels": np.array([]),
                    "scores": np.array([]),
                }
                for _ in predictions
            ]
        else:
            preds_dict = [
                {
                    "boxes": np.array(
                        [
                            [bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h]
                            for bbox in predictions[:4]
                        ],
                        dtype=np.float32,
                    ),
                    "labels": np.array(
                        [int(labels[5]) for labels in predictions], dtype=np.int64
                    ),
                    "scores": np.array(
                        [float(scores[4]) for scores in predictions], dtype=np.float32
                    ),
                }
            ]
        batch.predictions.set(preds_dict, self.predictions_spec)  # type: ignore


def _nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def _get_max_conf_and_id(confs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)
    return max_conf, max_id


def _process_batch(
    box_array: np.ndarray,
    max_conf: np.ndarray,
    max_id: np.ndarray,
    conf_thresh: float,
    num_classes: int,
    nms_thresh: float,
):
    bboxes_batch = []
    for i in range(box_array.shape[0]):
        bboxes_batch.append(
            _process_single(
                box_array[i],
                max_conf[i],
                max_id[i],
                conf_thresh,
                num_classes,
                nms_thresh,
            )
        )
    return bboxes_batch


def _process_single(
    box_array_single: np.ndarray,
    max_conf_single: np.ndarray,
    max_id_single: np.ndarray,
    conf_thresh: float,
    num_classes: int,
    nms_thresh: float,
):
    argwhere = max_conf_single > conf_thresh
    l_box_array = box_array_single[argwhere, :]
    l_max_conf = max_conf_single[argwhere]
    l_max_id = max_id_single[argwhere]

    bboxes = []
    for j in range(num_classes):
        bboxes.extend(_nms_for_class(l_box_array, l_max_conf, l_max_id, j, nms_thresh))
    return bboxes


def _nms_for_class(l_box_array, l_max_conf, l_max_id, cls, nms_thresh):
    cls_argwhere = l_max_id == cls
    ll_box_array = l_box_array[cls_argwhere, :]
    ll_max_conf = l_max_conf[cls_argwhere]
    ll_max_id = l_max_id[cls_argwhere]

    keep = _nms_cpu(ll_box_array, ll_max_conf, nms_thresh)

    bboxes = []
    if keep.size > 0:
        ll_box_array = ll_box_array[keep, :]
        ll_max_conf = ll_max_conf[keep]
        ll_max_id = ll_max_id[keep]

        for k in range(ll_box_array.shape[0]):
            bboxes.append(
                [
                    ll_box_array[k, 0],
                    ll_box_array[k, 1],
                    ll_box_array[k, 2],
                    ll_box_array[k, 3],
                    ll_max_conf[k],
                    ll_max_id[k],
                ]
            )
    return bboxes


def _post_processing(
    output: List, conf_thresh: float = 0.4, nms_thresh: float = 0.6
) -> List:
    box_array = to_numpy(output[0])
    confs = to_numpy(output[1])

    num_classes = confs.shape[2]
    box_array = box_array[:, :, 0]

    max_conf, max_id = _get_max_conf_and_id(confs)
    return _process_batch(
        box_array, max_conf, max_id, conf_thresh, num_classes, nms_thresh
    )
