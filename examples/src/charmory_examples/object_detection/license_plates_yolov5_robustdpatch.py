from pprint import pprint

import albumentations as A
from art.attacks.evasion import RobustDPatch
from art.estimators.object_detection import PyTorchYolo
from charmory_examples.utils.args import create_parser
import datasets
import numpy as np
import torch
from torchvision.ops import box_convert
import yolov5
from yolov5.utils.loss import ComputeLoss

from armory.art_experimental.attacks.patch import AttackWrapper
from armory.metrics.compute import BasicProfiler
from charmory.data import ArmoryDataLoader
from charmory.engine import EvaluationEngine
import charmory.evaluation as ev
from charmory.tasks.object_detection import ObjectDetectionTask
from charmory.track import track_init_params, track_params


def get_cli_args():
    parser = create_parser(
        description="Run YOLOv5m object detection for license plates",
        batch_size=4,
        export_every_n_batches=5,
        num_batches=20,
    )
    return parser.parse_args()


class Yolo(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.compute_loss = ComputeLoss(self.model.model.model)

    def forward(self, x, targets=None):
        if self.training:
            outputs = self.model.model.model(x)
            loss, _ = self.compute_loss(outputs, targets)
            return dict(loss_total=loss)
        return self.model(x)


@track_params
def load_model(
    model_path="keremberke/yolov5m-license-plate",
    conf=0.25,  # NMS confidence threshold
    iou=0.45,  # NMS IoU threshold
    agnostic=False,  # NMS class-agnostic
    multi_label=False,  # NMS multiple labels per box
    max_det=1000,  # maximum number of detections per image
):
    model = yolov5.load(model_path)
    # set model parameters
    model.conf = conf
    model.iou = iou
    model.agnostic = agnostic
    model.multi_label = multi_label
    model.max_det = max_det
    return Yolo(model)


def create_transform():
    augmentations = A.Compose(
        [
            A.LongestMaxSize(max_size=512),
            A.PadIfNeeded(
                min_height=512,
                min_width=512,
                border_mode=0,
                value=(0, 0, 0),
            ),
            A.ToFloat(max_value=255),  # Scale to [0,1]
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category", "id"],
        ),
    )

    def transform(sample):
        # pprint(sample)
        transformed = dict(**sample)
        transformed["image"] = []
        transformed["objects"] = []
        for idx in range(len(sample["image"])):
            res = augmentations(
                image=np.asarray(sample["image"][idx]),
                bboxes=sample["objects"][idx]["bbox"],
                category=sample["objects"][idx]["category"],
                id=sample["objects"][idx]["id"],
            )
            transformed["image"].append(res["image"].transpose(2, 0, 1))
            transformed["objects"].append(
                dict(
                    boxes=res["bboxes"],
                    labels=res["category"],
                    id=res["id"],
                )
            )
        for obj in transformed["objects"]:
            if len(obj.get("boxes", [])) > 0:
                obj["boxes"] = box_convert(
                    torch.tensor(obj["boxes"]), "xywh", "xyxy"
                ).numpy()
        return transformed

    return transform


@track_params
def main(batch_size, export_every_n_batches, num_batches):
    ###
    # Model
    ###
    model = load_model()
    detector = track_init_params(PyTorchYolo)(
        model,
        input_shape=(3, 512, 512),
        clip_values=(0, 1),
        attack_losses=("loss_total",),
    )

    ###
    # Dataset
    ###
    dataset = track_params(datasets.load_dataset)(
        "keremberke/license-plate-object-detection",
        name="full",
        split="test",
    )
    dataset.set_transform(create_transform())
    dataloader = ArmoryDataLoader(dataset, batch_size=batch_size, shuffle=True)

    ###
    # Attack
    ###
    patch = track_init_params(RobustDPatch)(
        detector,
        patch_shape=(3, 50, 50),
        patch_location=(231, 231),  # middle of 512x512
        batch_size=batch_size,
        sample_size=10,
        learning_rate=0.01,
        max_iter=20,
        targeted=False,
        verbose=False,
    )
    attack = AttackWrapper(patch)

    ###
    # Evaluation
    ###
    evaluation = ev.Evaluation(
        name="yolo-license-plate",
        description="YOLOv5m object detection for license plates",
        author="TwoSix",
        dataset=ev.Dataset(
            name="Vehicle Registration Plates Dataset",
            x_key="image",
            y_key="objects",
            test_dataloader=dataloader,
        ),
        model=ev.Model(
            name="YOLOv5m",
            model=detector,
        ),
        attack=ev.Attack(
            name="RobustDPatch",
            attack=attack,
            use_label_for_untargeted=False,
        ),
        metric=ev.Metric(profiler=BasicProfiler()),
    )

    ###
    # Engine
    ###
    task = ObjectDetectionTask(
        evaluation,
        export_every_n_batches=export_every_n_batches,
        iou_threshold=0.45,
        score_threshold=0.25,
    )
    engine = EvaluationEngine(task, limit_test_batches=num_batches)

    ###
    # Execute
    ###
    pprint(engine.run())


if __name__ == "__main__":
    main(**vars(get_cli_args()))
