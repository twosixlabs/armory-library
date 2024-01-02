"""Definition for the COCO object detection evaluation"""

from typing import Dict, Iterable, Optional

import art.attacks.evasion
from art.estimators.object_detection import PyTorchObjectDetector
import datasets
import jatic_toolbox
from jatic_toolbox.interop.huggingface import HuggingFaceObjectDetectionDataset
import numpy as np
import torchmetrics.detection
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from armory.data import ArmoryDataLoader
from armory.evaluation import Dataset, Evaluation, Metric, Model
from armory.examples.utils.args import create_parser
from armory.experimental.patch import AttackWrapper
from armory.experimental.transforms import BboxFormat, create_object_detection_transform
from armory.metrics.compute import BasicProfiler
from armory.model.object_detection import YolosTransformer
from armory.perturbation import ArtEvasionAttack, Perturbation
from armory.tasks.object_detection import ObjectDetectionTask
from armory.track import track_init_params, track_params


def get_cli_args(with_attack: bool):
    parser = create_parser(
        description="Run COCO object detection evaluation",
        batch_size=4,
        export_every_n_batches=5,
        num_batches=20,
    )
    if not with_attack:
        parser.add_argument(
            "dataset_path",
            type=str,
        )
    return parser.parse_args()


def _load_model():
    model = track_params(AutoModelForObjectDetection.from_pretrained)(
        pretrained_model_name_or_path="hustvl/yolos-tiny"
    )
    image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")

    transformer = YolosTransformer(model, image_processor)

    detector = track_init_params(PyTorchObjectDetector)(
        transformer,
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

    eval_model = Model(
        name="yolos",
        model=detector,
    )

    return eval_model, detector


def _load_dataset(batch_size: int, dataset_path: Optional[str] = None):
    if dataset_path is not None:
        ds = track_params(datasets.load_from_disk)(dataset_path=dataset_path)
        if isinstance(ds, datasets.DatasetDict):
            ds = ds["test"]
        dataset = HuggingFaceObjectDetectionDataset(ds, category_key="label")
    else:
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

    dataset.set_transform(
        create_object_detection_transform(
            max_size=512,
            float_max_value=255,
            format=BboxFormat.COCO,
            label_fields=["label", "id", "iscrowd"],
        )
    )

    dataloader = ArmoryDataLoader(dataset, batch_size=batch_size)

    return Dataset(
        name="coco",
        test_dataloader=dataloader,
        x_key="image",
        y_key="objects",
    )


def _create_metric():
    return Metric(
        profiler=BasicProfiler(),
        prediction={
            "map": torchmetrics.detection.MeanAveragePrecision(class_metrics=False),
        },
    )


def _create_attack(detector: PyTorchObjectDetector):
    patch = track_init_params(art.attacks.evasion.RobustDPatch)(
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

    return ArtEvasionAttack(
        name="RobustDPatch",
        attack=AttackWrapper(patch),
        use_label_for_untargeted=False,
    )


def create_evaluation(
    batch_size: int,
    with_attack: bool,
    dataset_path: Optional[str] = None,
    **kwargs,
) -> Evaluation:
    model, detector = _load_model()
    dataset = _load_dataset(batch_size=batch_size, dataset_path=dataset_path)
    perturbations: Dict[str, Iterable[Perturbation]] = dict(benign=[])
    if with_attack:
        perturbations["attack"] = [_create_attack(detector)]

    attack_type = "generated" if with_attack else "precomputed"

    evaluation = Evaluation(
        name=f"coco-yolos-{attack_type}-robustdpatch",
        description=f"COCO object detection with {attack_type} RobustDPatch attack",
        author="TwoSix",
        dataset=dataset,
        model=model,
        perturbations=perturbations,
        metric=_create_metric(),
    )

    return evaluation


@track_params
def create_evaluation_task(
    export_every_n_batches: int, with_attack: bool, **kwargs
) -> ObjectDetectionTask:
    evaluation = create_evaluation(with_attack=with_attack, **kwargs)

    task = ObjectDetectionTask(
        evaluation,
        export_every_n_batches=export_every_n_batches,
        class_metrics=False,
        skip_attack=not with_attack,
    )

    return task
