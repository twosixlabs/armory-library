"""
Example programmatic entrypoint for scenario execution
"""
import sys

import art.attacks.evasion
from art.estimators.classification import PyTorchClassifier
from jatic_toolbox import __version__ as jatic_version
from jatic_toolbox import load_dataset as load_jatic_dataset
from jatic_toolbox import load_model as load_jatic_model
import torch
import torch.nn as nn

from armory.instrument.config import MetricsLogger
from armory.metrics.compute import BasicProfiler
import armory.version
from charmory.data import ArmoryDataLoader, JaticImageClassificationDataset
from charmory.engine import LightningEngine
from charmory.evaluation import Attack, Dataset, Evaluation, Metric, Model, SysConfig
from charmory.experimental.example_results import print_outputs
from charmory.model.image_classification import JaticImageClassificationModel
from charmory.tasks.image_classification import ImageClassificationTask
from charmory.track import track_init_params, track_params
from charmory.utils import PILtoNumpy_HuggingFace_Variable_Length

BATCH_SIZE = 16


# Loads Imagenet 1k Classification HuggingFace Example


def load_huggingface_dataset():
    transform = PILtoNumpy_HuggingFace_Variable_Length(size=(500, 500))
    train_dataset = load_jatic_dataset(
        provider="huggingface",
        dataset_name="imagenet-1k",
        task="image-classification",
        split="train",
        use_auth_token=True,
    )

    train_dataset.set_transform(transform)

    train_dataloader = ArmoryDataLoader(
        dataset=JaticImageClassificationDataset(train_dataset),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_dataset = load_jatic_dataset(
        provider="huggingface",
        dataset_name="imagenet-1k",
        task="image-classification",
        split="test",
        use_auth_token=True,
    )
    test_dataset.set_transform(transform)
    test_dataloader = ArmoryDataLoader(
        dataset=JaticImageClassificationDataset(test_dataset),
        batch_size=BATCH_SIZE,
    )
    return train_dataloader, test_dataloader


def load_torchvision_model():
    print("Loading torchvision model from jatic_toolbox")
    model = track_params(load_jatic_model)(
        provider="torchvision",
        model_name="resnet34",
        task="image-classification",
    )

    classifier = track_init_params(PyTorchClassifier)(
        JaticImageClassificationModel(model),
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
        input_shape=(224, 224, 3),
        channels_first=False,
        nb_classes=1000,
        clip_values=(0.0, 1.0),
    )

    return classifier


def main(argv: list = sys.argv[1:]):
    if len(argv) > 0:
        if "--version" in argv:
            print(f"armory: {armory.version.__version__}")
            print(f"JATIC-toolbox: {jatic_version}")
            sys.exit(0)

    print("Armory: Example Programmatic Entrypoint for Scenario Execution")

    image_net_model = load_torchvision_model()
    model = Model(
        name="ImageNet1k",
        model=image_net_model,
    )

    train_dataset, test_dataset = load_huggingface_dataset()
    dataset = Dataset(
        name="imagenet", train_dataset=train_dataset, test_dataset=test_dataset
    )

    ###
    # The rest of this file was directly copied from the existing cifar example
    ###

    attack = Attack(
        name="PGD",
        attack=track_init_params(art.attacks.evasion.ProjectedGradientDescent)(
            image_net_model,
            batch_size=1,
            eps=0.031,
            eps_step=0.007,
            max_iter=20,
            num_random_init=1,
            random_eps=False,
            targeted=False,
            verbose=False,
        ),
        use_label_for_untargeted=True,
    )

    metric = Metric(
        profiler=BasicProfiler(),
        logger=MetricsLogger(
            supported_metrics=["accuracy"],
            perturbation=["linf"],
            task=["categorical_accuracy"],
            means=True,
            record_metric_per_sample=False,
        ),
    )

    sysconfig = SysConfig(gpus=["all"], use_gpu=True)

    evaluation = Evaluation(
        name="imagenet1k",
        description="Baseline imagenet1k image classification",
        author="msw@example.com",
        dataset=dataset,
        model=model,
        attack=attack,
        metric=metric,
        sysconfig=sysconfig,
    )

    task = ImageClassificationTask(
        evaluation, num_classes=1000, export_every_n_batches=5
    )

    engine = LightningEngine(task, limit_test_batches=5)
    results = engine.run()
    print_outputs(dataset, model, results)

    print("Imagenet 1k Experiment Complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
