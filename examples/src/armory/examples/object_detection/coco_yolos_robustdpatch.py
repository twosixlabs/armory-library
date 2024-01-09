from pprint import pprint

import albumentations as A
import art.attacks.evasion
from art.estimators.object_detection import PyTorchObjectDetector
import jatic_toolbox
import numpy as np
import torch
import torch.utils.data.dataloader
import torchmetrics.detection
from torchvision.transforms.v2 import GaussianBlur
from transformers import AutoImageProcessor, AutoModelForObjectDetection

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
from charmory.evaluation import Dataset, Evaluation

# from charmory.experimental.transforms import (
#     BboxFormat,
#     create_object_detection_transform,
# )
from charmory.export.object_detection import ObjectDetectionExporter
from charmory.metric import PerturbationMetric, PredictionMetric
from charmory.metrics.perturbation import PerturbationNormMetric
from charmory.model.object_detection import YolosTransformer
from charmory.perturbation import ArtEvasionAttack, CallablePerturbation
from charmory.track import track_init_params, track_params


def get_cli_args():
    parser = create_parser(
        description="Run COCO object detection example using models and datasets from the JATIC toolbox",
        batch_size=4,
        export_every_n_batches=5,
        num_batches=20,
    )
    return parser.parse_args()


@track_params(prefix="main")
def main(batch_size, export_every_n_batches, num_batches):
    ###
    # Model
    ###
    model = track_params(AutoModelForObjectDetection.from_pretrained)(
        pretrained_model_name_or_path="hustvl/yolos-tiny"
    )
    image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")

    transformer = YolosTransformer("faster-rcnn-resnet-50", model, image_processor)

    detector = track_init_params(PyTorchObjectDetector)(
        transformer,
        channels_first=False,
        input_shape=(512, 512, 3),
        clip_values=(0.0, 1.0),
        attack_losses=(
            "cardinality_error",
            "loss_bbox",
            "loss_ce",
            "loss_giou",
        ),
    )

    ###
    # Dataset
    ###
    dataset = track_params(jatic_toolbox.load_dataset)(
        provider="huggingface",
        dataset_name="rafaelpadilla/coco2017",
        task="object-detection",
        split="val",
        category_key="label",
    )

    # Have to filter out non-RGB images
    def filter(sample):
        shape = np.asarray(sample["image"]).shape
        return len(shape) == 3 and shape[2] == 3

    print(f"Dataset length prior to filtering: {len(dataset)}")
    dataset._dataset = dataset._dataset.filter(filter)
    print(f"Dataset length after filtering: {len(dataset)}")

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
            label_fields=["label", "id", "iscrowd"],
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
                label=sample["objects"][idx]["label"],
                id=sample["objects"][idx]["id"],
                iscrowd=sample["objects"][idx]["iscrowd"],
            )
            tmp["image"].append(res["image"])
            tmp["objects"].append(
                {
                    "bbox": res["bboxes"],
                    "label": res["label"],
                    "id": res["id"],
                    "iscrowd": res["iscrowd"],
                }
            )
        return tmp

    dataset.set_transform(transform)

    # dataset.set_transform(
    #     create_object_detection_transform(
    #         # Resize and pad images to 512x512
    #         max_size=512,
    #         # Scale to [0,1]
    #         float_max_value=255,
    #         to_tensor=True,
    #         format=BboxFormat.COCO,
    #         label_fields=["label", "id", "iscrowd"],
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
    # Perturbations
    ###

    blur = track_init_params(GaussianBlur)(
        kernel_size=5,
    )

    blur_perturb = CallablePerturbation(
        name="blur",
        perturbation=blur,
        inputs_accessor=Images.as_torch(
            dim=ImageDimensions.CHW,
            # scale=Scale(dtype=DataType.FLOAT, max=1.0),
        ),
    )

    patch = track_init_params(art.attacks.evasion.RobustDPatch)(
        detector,
        patch_shape=(50, 50, 3),
        patch_location=(231, 231),  # middle of 512x512
        batch_size=1,
        sample_size=10,
        learning_rate=0.01,
        max_iter=20,
        targeted=False,
        verbose=False,
    )

    patch_attack = ArtEvasionAttack(
        name="RobustDPatch",
        attack=AttackWrapper(patch),
        use_label_for_untargeted=False,
        inputs_accessor=Images.as_numpy(
            dim=ImageDimensions.HWC,
            scale=Scale(
                dtype=DataType.FLOAT,
                max=1.0,
                mean=YolosTransformer.DEFAULT_MEAN,
                std=YolosTransformer.DEFAULT_STD,
            ),
            dtype=np.float32,
        ),
    )

    ###
    # Metrics
    ###

    metrics = {
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
    }

    ###
    # Evaluation
    ###
    eval_dataset = Dataset(
        name="coco",
        dataloader=dataloader,
    )

    evaluation = Evaluation(
        name="coco-yolos-object-detection",
        description="COCO object detection using YOLO from HuggingFace",
        author="",
        dataset=eval_dataset,
        model=transformer,
        perturbations={
            "benign": [],
            "attack": [patch_attack],
            "blur": [blur_perturb],
            "attack_blur": [patch_attack, blur_perturb],
            "blur_attack": [blur_perturb, patch_attack],
        },
        metrics=metrics,
        exporter=ObjectDetectionExporter(),
        profiler=BasicProfiler(),
    )

    ###
    # Engine
    ###

    engine = EvaluationEngine(
        evaluation,
        limit_test_batches=num_batches,
        export_every_n_batches=export_every_n_batches,
    )
    results = engine.run()

    pprint(results)


if __name__ == "__main__":
    main(**vars(get_cli_args()))
