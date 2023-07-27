"""
Example programmatic entrypoint for scenario execution

This file is unnecessarily complicated for the sake of demonstrating
interoperability between dataset and model providers. This file is NOT
a good example of using the JATIC toolbox or Armory.
"""
import argparse
import json
from pprint import pprint
import sys

import art.attacks.evasion
from art.estimators.classification import PyTorchClassifier
from jatic_toolbox import __version__ as jatic_version
from jatic_toolbox import load_dataset as load_jatic_dataset
from jatic_toolbox import load_model as load_jatic_model
import torch
import torch.nn as nn

import armory.baseline_models.pytorch.resnet18
import armory.data.datasets
import armory.version
from charmory.data import JaticVisionDatasetGenerator
from charmory.engine import Engine
from charmory.evaluation import (
    Attack,
    Dataset,
    Evaluation,
    Metric,
    Model,
    Scenario,
    SysConfig,
)
import charmory.scenarios.image_classification
from charmory.utils import (
    adapt_jatic_image_classification_model_for_art,
    create_jatic_image_classification_dataset_transform,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
TRAINING_EPOCHS = 20


def load_huggingface_dataset(transform):
    print("Loading HuggingFace dataset from jatic_toolbox")

    train_dataset = load_jatic_dataset(
        provider="huggingface",
        dataset_name="cifar10",
        task="image-classification",
        split="train",
    )
    train_dataset.set_transform(transform)
    train_dataset_generator = JaticVisionDatasetGenerator(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        epochs=TRAINING_EPOCHS,
        shuffle=True,
        size=512,  # Use a subset just for demo purposes
    )

    test_dataset = load_jatic_dataset(
        provider="huggingface",
        dataset_name="cifar10",
        task="image-classification",
        split="test",
    )
    test_dataset.set_transform(transform)
    test_dataset_generator = JaticVisionDatasetGenerator(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        epochs=1,
        shuffle=False,
        size=512,  # Use a subset just for demo purposes
    )

    return train_dataset_generator, test_dataset_generator


def load_torchvision_dataset(transform):
    print("Loading torchvision dataset from jatic_toolbox")
    train_dataset = load_jatic_dataset(
        provider="torchvision",
        dataset_name="CIFAR10",
        task="image-classification",
        split="train",
        root="/tmp/torchvision_datasets",
        download=True,
    )
    train_dataset.set_transform(transform)
    train_dataset_generator = JaticVisionDatasetGenerator(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        epochs=TRAINING_EPOCHS,
        shuffle=True,
        size=512,  # Use a subset just for demo purposes
    )

    test_dataset = load_jatic_dataset(
        provider="torchvision",
        dataset_name="CIFAR10",
        task="image-classification",
        split="test",
        root="/tmp/torchvision_datasets",
        download=True,
    )
    test_dataset.set_transform(transform)
    test_dataset_generator = JaticVisionDatasetGenerator(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        epochs=1,
        shuffle=False,
        size=512,  # Use a subset just for demo purposes
    )

    return train_dataset_generator, test_dataset_generator


def load_huggingface_model():
    print("Loading HuggingFace model from jatic_toolbox")
    model = load_jatic_model(
        provider="huggingface",
        model_name="microsoft/resnet-18",
        task="image-classification",
    )
    adapt_jatic_image_classification_model_for_art(model)
    model.to(DEVICE)

    classifier = PyTorchClassifier(
        model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
        input_shape=(224, 224, 3),
        channels_first=False,
        nb_classes=10,
        clip_values=(0.0, 1.0),
    )

    transform = create_jatic_image_classification_dataset_transform(model.preprocessor)

    return classifier, transform


def load_torchvision_model():
    print("Loading torchvision model from jatic_toolbox")
    model = load_jatic_model(
        provider="torchvision",
        model_name="resnet18",
        task="image-classification",
    )
    adapt_jatic_image_classification_model_for_art(model)
    model.to(DEVICE)

    classifier = PyTorchClassifier(
        model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
        input_shape=(224, 224, 3),
        channels_first=False,
        nb_classes=10,
        clip_values=(0.0, 1.0),
    )

    transform = create_jatic_image_classification_dataset_transform(model.preprocessor)

    return classifier, transform


def main():
    parser = argparse.ArgumentParser(
        description="Run example using models and datasets from the JATIC toolbox",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["huggingface", "torchvision"],
        default="huggingface",
        help="Source of CIFAR10 dataset",
    )
    parser.add_argument(
        "--model",
        choices=["huggingface", "torchvision"],
        default="huggingface",
        help="Source of ResNet-50 model",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"armory: {armory.version.__version__}\nJATIC-toolbox: {jatic_version}",
    )
    args = parser.parse_args()

    print("Armory: Example Programmatic Entrypoint for Scenario Execution")

    if args.model == "torchvision":
        loaded_model, transform = load_torchvision_model()
    else:
        loaded_model, transform = load_huggingface_model()

    if args.dataset == "torchvision":
        train_dataset, test_dataset = load_torchvision_dataset(transform)
    else:
        train_dataset, test_dataset = load_huggingface_dataset(transform)

    dataset = Dataset(
        name="CIFAR10",
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )

    model = Model(
        name="ResNet-18",
        model=loaded_model,
    )

    ###
    # The rest of this file was directly copied from the existing cifar example
    ###

    attack = Attack(
        function=art.attacks.evasion.ProjectedGradientDescent,
        kwargs={
            "batch_size": 1,
            "eps": 0.031,
            "eps_step": 0.007,
            "max_iter": 20,
            "num_random_init": 1,
            "random_eps": False,
            "targeted": False,
            "verbose": False,
        },
        knowledge="white",
        use_label=True,
        type=None,
    )

    scenario = Scenario(
        function=charmory.scenarios.image_classification.ImageClassificationTask,
        kwargs={},
    )

    metric = Metric(
        profiler_type="basic",
        supported_metrics=["accuracy"],
        perturbation=["linf"],
        task=["categorical_accuracy"],
        means=True,
        record_metric_per_sample=False,
    )

    sysconfig = SysConfig(gpus=["all"], use_gpu=True)

    baseline = Evaluation(
        name="cifar_baseline",
        description="Baseline cifar10 image classification",
        author="msw@example.com",
        dataset=dataset,
        model=model,
        attack=attack,
        scenario=scenario,
        metric=metric,
        sysconfig=sysconfig,
    )

    print(f"Starting Demo for {baseline.name}")

    cifar_engine = Engine(baseline)
    cifar_engine.train(nb_epochs=TRAINING_EPOCHS)
    results = cifar_engine.run()

    print("=" * 64)
    pprint(baseline)
    print("-" * 64)
    print(
        json.dumps(
            results, default=lambda o: "<not serializable>", indent=4, sort_keys=True
        )
    )

    print("=" * 64)
    print(dataset.train_dataset)
    print(dataset.test_dataset)
    print("-" * 64)
    print(model)

    print("=" * 64)
    print("JATIC Experiment Complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
