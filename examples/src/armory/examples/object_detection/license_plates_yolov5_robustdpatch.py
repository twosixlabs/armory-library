from pprint import pprint

import albumentations as A
from art.attacks.evasion import RobustDPatch
from art.estimators.object_detection import PyTorchYolo
import datasets
import numpy as np
import torch
import torchmetrics.detection
import yolov5

from armory.art_experimental.attacks.patch import AttackWrapper
from armory.examples.utils.args import create_parser
from armory.metrics.compute import BasicProfiler
from charmory.data import (
    BBoxFormat,
    BoundingBoxes,
    DataType,
    ImageDimensions,
    Images,
    Scale,
)
from charmory.dataset import ObjectDetectionDataLoader
from charmory.engine import EvaluationEngine
import charmory.evaluation as ev

# from charmory.experimental.transforms import (
#     BboxFormat,
#     create_object_detection_transform,
# )
from charmory.export.object_detection import ObjectDetectionExporter
from charmory.metric import PerturbationMetric, PredictionMetric
from charmory.metrics.perturbation import PerturbationNormMetric
from charmory.model.object_detection import YoloV5ObjectDetector
from charmory.perturbation import ArtEvasionAttack
from charmory.track import track_init_params, track_params


def get_cli_args():
    parser = create_parser(
        description="Run YOLOv5m object detection for license plates",
        batch_size=4,
        export_every_n_batches=5,
        num_batches=20,
    )
    return parser.parse_args()


@track_params
def main(batch_size, export_every_n_batches, num_batches):
    ###
    # Model
    ###
    model = YoloV5ObjectDetector(
        name="YOLOv5m",
        model=yolov5.load("keremberke/yolov5m-license-plate"),
    )
    detector = track_init_params(PyTorchYolo)(
        model,
        input_shape=(512, 512, 3),
        channels_first=False,
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
    resize = A.Compose(
        [
            A.LongestMaxSize(512),
            A.PadIfNeeded(
                min_height=512,
                min_width=512,
                border_mode=0,
                value=(0, 0, 0),
            ),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category", "id"],
        ),
    )

    def transform(sample):
        tmp = dict(**sample)
        tmp["image"] = []
        tmp["objects"] = []
        for idx, image in enumerate(sample["image"]):
            res = resize(
                image=np.asarray(image),
                bboxes=sample["objects"][idx]["bbox"],
                category=sample["objects"][idx]["category"],
                id=sample["objects"][idx]["id"],
            )
            tmp["image"].append(res["image"])
            tmp["objects"].append(
                {
                    "boxes": res["bboxes"],
                    "labels": res["category"],
                    "id": res["id"],
                }
            )
        return tmp

    dataset.set_transform(transform)
    # dataset.set_transform(
    #     create_object_detection_transform(
    #         max_size=512,
    #         format=BboxFormat.XYWH,
    #         label_fields=["category", "id"],
    #         rename_object_fields={"bbox": "boxes", "category": "labels"},
    #     )
    # )
    dataloader = ObjectDetectionDataLoader(
        dataset,
        format=BBoxFormat.XYWH,
        boxes_key="boxes",
        dim=ImageDimensions.HWC,
        scale=Scale(dtype=DataType.UINT8, max=255),
        image_key="image",
        labels_key="labels",
        objects_key="objects",
        batch_size=batch_size,
        shuffle=True,
    )

    ###
    # Attack
    ###
    patch = track_init_params(RobustDPatch)(
        detector,
        patch_shape=(50, 50, 3),
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
            dataloader=dataloader,
        ),
        model=model,
        perturbations=dict(
            benign=[],
            attack=[
                ArtEvasionAttack(
                    name="RobustDPatch",
                    attack=attack,
                    use_label_for_untargeted=False,
                    inputs_accessor=Images.as_numpy(
                        dim=ImageDimensions.HWC,
                        scale=Scale(dtype=DataType.FLOAT, max=1.0),
                        dtype=np.float32,
                    ),
                )
            ],
        ),
        metrics={
            "linf_norm": PerturbationMetric(
                PerturbationNormMetric(ord=torch.inf),
                Images.as_torch(
                    dim=ImageDimensions.HWC, scale=Scale(dtype=DataType.FLOAT, max=1.0)
                ),
            ),
            "map": PredictionMetric(
                torchmetrics.detection.MeanAveragePrecision(class_metrics=False),
                BoundingBoxes.as_torch(format=BBoxFormat.XYXY),
            ),
        },
        exporter=ObjectDetectionExporter(),
        profiler=BasicProfiler(),
    )

    ###
    # Engine
    ###
    engine = EvaluationEngine(
        evaluation,
        export_every_n_batches=export_every_n_batches,
        limit_test_batches=num_batches,
    )

    ###
    # Execute
    ###
    pprint(engine.run())
    pprint(engine.metrics.compute())


if __name__ == "__main__":
    main(**vars(get_cli_args()))
