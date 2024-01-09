from pathlib import Path
from pprint import pprint
import sys

from PIL import Image
import art.attacks.evasion
from art.estimators.object_detection import PyTorchFasterRCNN
import boto3
import botocore
from jatic_toolbox import __version__ as jatic_version
from jatic_toolbox.interop.huggingface import HuggingFaceObjectDetectionDataset
from jatic_toolbox.interop.torchvision import TorchVisionObjectDetector
import torch
import torchmetrics.detection
from torchvision.transforms._presets import ObjectDetection

from armory.data import ArmoryDataLoader
from armory.engine import EvaluationEngine
from armory.evaluation import Dataset, Evaluation, Metric, Model
from armory.experimental.patch import AttackWrapper
from armory.experimental.transforms import BboxFormat, create_object_detection_transform
from armory.metrics.compute import BasicProfiler
from armory.model.object_detection import JaticObjectDetectionModel
from armory.perturbation import ArtEvasionAttack
from armory.tasks.object_detection import ObjectDetectionTask
import armory.version

torch.set_float32_matmul_precision("high")
from datasets import load_from_disk
from datasets.filesystems import S3FileSystem

import armory.data.datasets
from armory.track import track_init_params
from armory.utils import create_jatic_dataset_transform

BATCH_SIZE = 1
TRAINING_EPOCHS = 20
BUCKET_NAME = "armory-library-data"
KEY = "fasterrcnn_mobilenet_v3_2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_huggingface_dataset():
    s3 = S3FileSystem(anon=False)
    train_dataset = load_from_disk("s3://armory-library-data/datasets/train/", fs=s3)

    new_dataset = train_dataset.train_test_split(test_size=0.4, seed=3)
    train_dataset, test_dataset = new_dataset["train"], new_dataset["test"]

    train_dataset, test_dataset = HuggingFaceObjectDetectionDataset(
        train_dataset
    ), HuggingFaceObjectDetectionDataset(test_dataset)

    return train_dataset, test_dataset


def main(argv: list = sys.argv[1:]):
    if len(argv) > 0:
        if "--version" in argv:
            print(f"armory: {armory.version.__version__}")
            print(f"JATIC-toolbox: {jatic_version}")
            sys.exit(0)

    print("Armory: Example Programmatic Entrypoint for Scenario Execution")
    ###
    # Model
    ###
    s3 = boto3.resource("s3")
    my_file = Path.cwd() / "fasterrcnn_mobilenet_v3_2"
    if not my_file.is_file():
        try:
            s3.Bucket(BUCKET_NAME).download_file(KEY, my_file)
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                print("The object does not exist.")
            else:
                raise

    model = torch.load(my_file)
    model.to(DEVICE)

    model = TorchVisionObjectDetector(
        model=model, processor=ObjectDetection(), labels=None
    )
    model.forward = model._model.forward

    detector = track_init_params(PyTorchFasterRCNN)(
        JaticObjectDetectionModel(model),
        channels_first=True,
        clip_values=(0.0, 1.0),
    )

    model_transform = create_jatic_dataset_transform(model.preprocessor)

    train_dataset, test_dataset = load_huggingface_dataset()

    transform = create_object_detection_transform(
        max_size=500,
        format=BboxFormat.XYXY,
        label_fields=["category"],
        image_from_np=Image.fromarray,
        postprocessor=model_transform,
    )

    train_dataset.set_transform(transform)
    test_dataset.set_transform(transform)

    train_dataloader = ArmoryDataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
    )
    test_dataloader = ArmoryDataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
    )
    eval_dataset = Dataset(
        name="XVIEW",
        x_key="image",
        y_key="objects",
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
    )
    eval_model = Model(
        name="xview-trained-fasterrcnn-resnet-50",
        model=detector,
    )

    patch = track_init_params(art.attacks.evasion.RobustDPatch)(
        detector,
        patch_shape=(3, 32, 32),
        batch_size=BATCH_SIZE,
        max_iter=20,
        targeted=False,
        verbose=False,
    )

    eval_attack = ArtEvasionAttack(
        name="RobustDPatch",
        attack=AttackWrapper(patch),
        use_label_for_untargeted=False,
    )

    eval_metric = Metric(
        profiler=BasicProfiler(),
        prediction={
            "map": torchmetrics.detection.MeanAveragePrecision(class_metrics=False),
        },
    )

    evaluation = Evaluation(
        name="xview-object-detection",
        description="XView object detection from HuggingFace",
        author="Chris Honaker",
        dataset=eval_dataset,
        model=eval_model,
        perturbations={
            "benign": [],
            "attack": [eval_attack],
        },
        metric=eval_metric,
    )

    ###
    # Engine
    ###

    task = ObjectDetectionTask(
        evaluation,
        export_every_n_batches=2,
        class_metrics=False,
    )
    engine = EvaluationEngine(task, limit_test_batches=10)
    results = engine.run()

    pprint(results)


if __name__ == "__main__":
    sys.exit(main())
