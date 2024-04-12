
from pytorch_yolo.utils import nms_cpu
import numpy as np

############################################
######### POST PROCESSING #########
############################################


def to_numpy(tensor_obj):
    if type(tensor_obj).__name__ != "ndarray":
        return tensor_obj.cpu().detach().numpy()
    return tensor_obj


def get_max_conf_and_id(confs):
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)
    return max_conf, max_id


def process_batch(
    box_array, max_conf, max_id, conf_thresh, num_classes, nms_thresh
):
    bboxes_batch = []
    for i in range(box_array.shape[0]):
        bboxes_batch.append(
            process_single(
                box_array[i],
                max_conf[i],
                max_id[i],
                conf_thresh,
                num_classes,
                nms_thresh,
            )
        )
    return bboxes_batch


def process_single(
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
        bboxes.extend(
            nms_for_class(l_box_array, l_max_conf, l_max_id, j, nms_thresh)
        )
    return bboxes


def nms_for_class(l_box_array, l_max_conf, l_max_id, cls, nms_thresh):
    cls_argwhere = l_max_id == cls
    ll_box_array = l_box_array[cls_argwhere, :]
    ll_max_conf = l_max_conf[cls_argwhere]
    ll_max_id = l_max_id[cls_argwhere]

    keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)

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


def post_processing(
    output: list, conf_thresh: float = 0.4, nms_thresh: float = 0.6
) -> list:
    box_array = to_numpy(output[0])
    confs = to_numpy(output[1])

    num_classes = confs.shape[2]
    box_array = box_array[:, :, 0]

    max_conf, max_id = get_max_conf_and_id(confs)
    return process_batch(
        box_array, max_conf, max_id, conf_thresh, num_classes, nms_thresh
    )
