"""
Example Armory evaluation of license plate object detection with YOLOS against a
DPatch attack
"""

from pprint import pprint

import art.attacks.evasion
import art.estimators.object_detection
import datasets
import torch
import torch.nn
import torchmetrics.detection
import torchvision.transforms.v2
import transformers

import armory.data
import armory.engine
import armory.evaluation
import armory.experimental.patch
import armory.experimental.transforms
import armory.metrics.compute
import armory.metrics.perturbation
import armory.model.object_detection
import armory.perturbation
import armory.tasks.object_detection
import armory.track


def parse_cli_args():
    """Parse command-line arguments"""
    from armory.examples.utils.args import create_parser

    parser = create_parser(
        description="Perform license plate object detection with YOLOS",
        batch_size=4,
        export_every_n_batches=5,
        num_batches=20,
    )
    return parser.parse_args()


def load_model():
    """Load YOLOS model from HuggingFace"""
    model_name = "nickmuchi/yolos-small-finetuned-license-plate-detection"

    hf_model = armory.track.track_params(
        transformers.AutoModelForObjectDetection.from_pretrained
    )(pretrained_model_name_or_path=model_name)

    hf_processor = transformers.AutoImageProcessor.from_pretrained(model_name)

    armory_model = armory.model.object_detection.YolosTransformer(
        hf_model, hf_processor
    )

    art_detector = armory.track.track_init_params(
        art.estimators.object_detection.PyTorchObjectDetector
    )(
        armory_model,
        channels_first=True,
        input_shape=(3, 512, 512),
        clip_values=(0.0, 1.0),
        attack_losses=(
            "cardinality_error",
            "loss_bbox",
            "loss_ce",
            "loss_giou",
        ),
    )

    evaluation_model = armory.evaluation.Model(
        name="YOLOS",
        model=art_detector,
    )

    return evaluation_model, art_detector


def load_dataset(batch_size: int, shuffle: bool):
    hf_dataset = datasets.load_dataset(
        "keremberke/license-plate-object-detection", name="full", split="test"
    )
    assert isinstance(hf_dataset, datasets.Dataset)

    od_transform = armory.experimental.transforms.create_object_detection_transform(
        # Resize and pad images to 512x512
        max_size=512,
        # Scale to [0,1]
        float_max_value=255,
        format=armory.experimental.transforms.BboxFormat.COCO,
        label_fields=["category", "id"],
        rename_object_fields={"bbox": "boxes", "category": "labels"},
    )

    def transform(sample):
        sample = od_transform(sample)
        # The dataset in HF uses label 0 for license plates, but the original
        # dataset from roboflow used label 1. Since the YOLOS model was
        # fine-tuned using label 1, we have to modify the ground truth labels
        # in order to get meaningful metrics.
        for obj in sample["objects"]:
            obj["labels"] = [1 for _ in obj["labels"]]
        return sample

    hf_dataset.set_transform(transform)

    dataloader = armory.data.ArmoryDataLoader(
        hf_dataset, batch_size=batch_size, shuffle=shuffle
    )

    evaluation_dataset = armory.evaluation.Dataset(
        name="Roboflow Vehicle Registration Plates Dataset",
        test_dataloader=dataloader,
        x_key="image",
        y_key="objects",
    )

    return evaluation_dataset


def create_attack(detector):
    dpatch = armory.track.track_init_params(art.attacks.evasion.RobustDPatch)(
        detector,
        patch_shape=(3, 50, 50),
        patch_location=(231, 231),  # middle of 512x512
        batch_size=1,
        sample_size=10,
        learning_rate=0.01,
        max_iter=20,
        targeted=False,
        verbose=False,
    )

    evaluation_attack = armory.perturbation.ArtEvasionAttack(
        name="RobustDPatch",
        attack=armory.experimental.patch.AttackWrapper(dpatch),
        use_label_for_untargeted=False,
    )

    return evaluation_attack


def create_blur():
    blur = armory.track.track_init_params(torchvision.transforms.v2.GaussianBlur)(
        kernel_size=5,
    )

    evaluation_perturbation = armory.perturbation.TorchTransformPerturbation(
        name="blur",
        perturbation=blur,
    )

    return evaluation_perturbation


def create_metric():
    evaluation_metric = armory.evaluation.Metric(
        profiler=armory.metrics.compute.BasicProfiler(),
        perturbation={
            "linf_norm": armory.metrics.perturbation.PerturbationNormMetric(
                ord=torch.inf
            ),
        },
        prediction={
            "map": torchmetrics.detection.MeanAveragePrecision(class_metrics=False),
        },
    )

    return evaluation_metric


@armory.track.track_params(prefix="main")
def main(batch_size, export_every_n_batches, num_batches, seed, shuffle):
    """Perform evaluation"""
    if seed is not None:
        torch.manual_seed(seed)

    model, art_detector = load_model()

    dataset = load_dataset(batch_size, shuffle)
    attack = create_attack(art_detector)
    metric = create_metric()

    evaluation = armory.evaluation.Evaluation(
        name="license-plate-detection-yolos",
        description="License plate object detection using yolos",
        author="TwoSix",
        dataset=dataset,
        model=model,
        perturbations={
            "benign": [],
            "attack": [attack],
        },
        metric=metric,
    )

    task = armory.tasks.object_detection.ObjectDetectionTask(
        evaluation, export_every_n_batches=export_every_n_batches
    )
    engine = armory.engine.EvaluationEngine(task, limit_test_batches=num_batches)
    results = engine.run()

    pprint(results)


if __name__ == "__main__":
    main(**vars(parse_cli_args()))
