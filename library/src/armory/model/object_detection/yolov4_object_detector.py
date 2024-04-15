from typing import Optional

import numpy as np
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
class YoloV4ObjectDetector(ObjectDetector):
    """
    Model wrapper with pre-applied output adapters for YOLOv4 models.

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
        )

    def forward(self, x, targets=None):
        """
        Invokes the wrapped model. If in training and given targets, then the
        loss is computed and returned rather than the raw predictions.
        """
        # TODO: (jxc) From YoloV5ObjectDetector, rework this for YoloV4ObjectDetector
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
        self.eval()
        inputs = self.inputs_accessor.get(batch.inputs)
        _, h, w = inputs.shape
        outputs = self(inputs)
        outputs = _post_processing(outputs)
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
                        [
                            [output[0] * w, output[1] * h, output[2] * w, output[3] * h]
                            for output in outputs[0][:4]
                        ],
                        dtype=np.float32,
                    ),
                    "labels": np.array(
                        [int(output[5]) for output in outputs[0]], dtype=np.int64
                    ),
                    "scores": np.array(
                        [float(output[4]) for output in outputs[0]], dtype=np.float32
                    ),
                }
                for output in outputs
            ]
            self.predictions_accessor.set(batch.predictions, outputs_dict)


def _nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    # print(boxes.shape)
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


def _to_numpy(tensor_obj):
    if type(tensor_obj).__name__ != "ndarray":
        return tensor_obj.cpu().detach().numpy()
    return tensor_obj


def _get_max_conf_and_id(confs):
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)
    return max_conf, max_id


def _process_batch(box_array, max_conf, max_id, conf_thresh, num_classes, nms_thresh):
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
    box_array_single,
    max_conf_single,
    max_id_single,
    conf_thresh,
    num_classes,
    nms_thresh,
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
    output: list, conf_thresh: float = 0.4, nms_thresh: float = 0.6
) -> list:
    box_array = _to_numpy(output[0])
    confs = _to_numpy(output[1])

    num_classes = confs.shape[2]
    box_array = box_array[:, :, 0]

    max_conf, max_id = _get_max_conf_and_id(confs)
    return _process_batch(
        box_array, max_conf, max_id, conf_thresh, num_classes, nms_thresh
    )
