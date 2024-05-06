"""
Example Armory evaluation of license plate object detection with YOLOv5 against
a DPatch attack
"""

from pprint import pprint

import albumentations as A
import albumentations.pytorch.transforms
import art.attacks.evasion
import art.estimators.object_detection
import datasets
import numpy as np
import torch
import torchmetrics.detection
import torchvision.transforms.v2
import yolov5

import armory.data
import armory.dataset
import armory.engine
import armory.evaluation
import armory.export.criteria
import armory.export.drise
import armory.export.object_detection
import armory.metric
import armory.metrics.compute
import armory.metrics.detection
import armory.metrics.perturbation
import armory.metrics.tide
import armory.model.object_detection
import armory.perturbation
import armory.track


def parse_cli_args():
    """Parse command-line arguments"""
    from armory.examples.utils.args import create_parser

    parser = create_parser(
        description="Perform license plate object detection",
        batch_size=4,
        export_every_n_batches=5,
        num_batches=20,
    )
    return parser.parse_args()


def load_model():
    """Load YOLOv5 model from HuggingFace"""

    hf_model = armory.track.track_params(yolov5.load)(
        model_path="keremberke/yolov5m-license-plate"
    )

    armory_model = armory.model.object_detection.YoloV5ObjectDetector(
        name="YOLOv5m",
        model=hf_model,
    )

    art_detector = armory.track.track_init_params(
        art.estimators.object_detection.PyTorchYolo
    )(
        armory_model,
        channels_first=True,
        input_shape=(3, 512, 512),
        clip_values=(0.0, 1.0),
        attack_losses=("loss_total",),
    )

    return armory_model, art_detector


def load_dataset(batch_size: int, shuffle: bool):
    hf_dataset = datasets.load_dataset(
        "keremberke/license-plate-object-detection", name="full", split="test"
    )
    assert isinstance(hf_dataset, datasets.Dataset)

    resize = A.Compose(
        [
            A.LongestMaxSize(512),
            A.PadIfNeeded(
                min_height=512,
                min_width=512,
                border_mode=0,
                value=(0, 0, 0),
            ),
            A.ToFloat(max_value=255),
            albumentations.pytorch.ToTensorV2(),
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
        for image, objects in zip(sample["image"], sample["objects"]):
            res = resize(
                image=np.asarray(image),
                bboxes=objects["bbox"],
                category=objects["category"],
                id=objects["id"],
            )
            tmp["image"].append(res.pop("image"))
            tmp["objects"].append(res)
        return tmp

    hf_dataset.set_transform(transform)

    dataloader = armory.dataset.ObjectDetectionDataLoader(
        hf_dataset,
        format=armory.data.BBoxFormat.XYWH,
        boxes_key="bboxes",
        dim=armory.data.ImageDimensions.CHW,
        scale=armory.data.Scale(dtype=armory.data.DataType.FLOAT, max=1.0),
        image_key="image",
        labels_key="category",
        objects_key="objects",
        batch_size=batch_size,
        shuffle=shuffle,
    )

    evaluation_dataset = armory.evaluation.Dataset(
        name="Roboflow Vehicle Registration Plates Dataset",
        dataloader=dataloader,
    )

    return evaluation_dataset


def create_attack(detector, batch_size: int = 1):
    dpatch = armory.track.track_init_params(art.attacks.evasion.RobustDPatch)(
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

    evaluation_attack = armory.perturbation.ArtPatchAttack(
        name="RobustDPatch",
        attack=dpatch,
        use_label_for_untargeted=False,
        generate_every_batch=True,
    )

    return evaluation_attack


def create_blur():
    blur = armory.track.track_init_params(torchvision.transforms.v2.GaussianBlur)(
        kernel_size=5,
    )

    evaluation_perturbation = armory.perturbation.CallablePerturbation(
        name="blur",
        perturbation=blur,
        inputs_spec=armory.data.TorchSpec(),
    )

    return evaluation_perturbation


def create_metrics():
    return {
        "linf_norm": armory.metric.PerturbationMetric(
            armory.metrics.perturbation.PerturbationNormMetric(ord=torch.inf),
        ),
        "map": armory.metric.PredictionMetric(
            torchmetrics.detection.MeanAveragePrecision(class_metrics=False),
            armory.data.TorchBoundingBoxSpec(format=armory.data.BBoxFormat.XYXY),
        ),
        "tide": armory.metrics.tide.TIDE.create(),
        "detection": armory.metrics.detection.ObjectDetectionRates.create(),
    }


def create_exporters(model, export_every_n_batches):
    """Create sample exporters"""
    return [
        armory.export.object_detection.ObjectDetectionExporter(
            criterion=armory.export.criteria.every_n_batches(export_every_n_batches)
        ),
        armory.export.drise.DRiseSaliencyObjectDetectionExporter(
            model,
            criterion=armory.export.criteria.every_n_batches(export_every_n_batches),
            num_classes=1,
            num_masks=10,
        ),
    ]


@armory.track.track_params(prefix="main")
def main(batch_size, export_every_n_batches, num_batches, seed, shuffle):
    """Perform evaluation"""
    if seed is not None:
        torch.manual_seed(seed)

    evaluation = armory.evaluation.Evaluation(
        name="license-plate-detection-yolov5",
        description="License plate object detection using yolov5",
        author="TwoSix",
    )

    # Model
    with evaluation.autotrack():
        model, art_detector = load_model()
    evaluation.use_model(model)

    # Dataset
    with evaluation.autotrack():
        dataset = load_dataset(batch_size, shuffle)
    evaluation.use_dataset(dataset)

    # Metrics/Exporters
    evaluation.use_metrics(create_metrics())
    evaluation.use_exporters(create_exporters(model, export_every_n_batches))

    # Chains
    with evaluation.add_chain("benign"):
        pass

    with evaluation.add_chain("attack") as chain:
        chain.add_perturbation(create_attack(art_detector, batch_size))

    engine = armory.engine.EvaluationEngine(
        evaluation,
        profiler=armory.metrics.compute.BasicProfiler(),
        limit_test_batches=num_batches,
    )
    results = engine.run()

    pprint(results)


if __name__ == "__main__":
    main(**vars(parse_cli_args()))
