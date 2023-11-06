"""
Example programmatic entrypoint for scenario execution

This file is unnecessarily complicated for the sake of demonstrating
interoperability between dataset and model providers. This file is NOT
a good example of using the JATIC toolbox or Armory.
"""
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
from armory.examples.utils.args import create_parser
from armory.metrics.compute import BasicProfiler
import armory.version
from charmory.data import ArmoryDataLoader
from charmory.engine import EvaluationEngine
from charmory.evaluation import Attack, Dataset, Evaluation, Metric, Model
from charmory.experimental.example_results import print_outputs
from charmory.model.image_classification import JaticImageClassificationModel
from charmory.tasks.image_classification import ImageClassificationTask
from charmory.track import track_init_params, track_params
from charmory.utils import create_jatic_dataset_transform


def load_huggingface_dataset(transform, batch_size):
    print("Loading HuggingFace dataset from jatic_toolbox")

    train_dataset = track_params(load_jatic_dataset)(
        provider="huggingface",
        dataset_name="cifar10",
        task="image-classification",
        split="train",
    )
    train_dataset.set_transform(transform)
    train_dataloader = ArmoryDataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_dataset = track_params(load_jatic_dataset)(
        provider="huggingface",
        dataset_name="cifar10",
        task="image-classification",
        split="test",
    )
    test_dataset.set_transform(transform)
    test_dataloader = ArmoryDataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_dataloader, test_dataloader


def load_torchvision_dataset(transform, batch_size):
    print("Loading torchvision dataset from jatic_toolbox")
    train_dataset = track_params(load_jatic_dataset)(
        provider="torchvision",
        dataset_name="CIFAR10",
        task="image-classification",
        split="train",
        root="/tmp/torchvision_datasets",
        download=True,
    )
    train_dataset.set_transform(transform)
    train_dataloader = ArmoryDataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_dataset = track_params(load_jatic_dataset)(
        provider="torchvision",
        dataset_name="CIFAR10",
        task="image-classification",
        split="test",
        root="/tmp/torchvision_datasets",
        download=True,
    )
    test_dataset.set_transform(transform)
    test_dataloader = ArmoryDataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_dataloader, test_dataloader


def load_huggingface_model():
    print("Loading HuggingFace model from jatic_toolbox")
    model = track_params(load_jatic_model)(
        provider="huggingface",
        model_name="jadohu/BEiT-finetuned",
        task="image-classification",
    )

    classifier = track_init_params(PyTorchClassifier)(
        JaticImageClassificationModel(model),
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
        input_shape=(224, 224, 3),
        channels_first=False,
        nb_classes=10,
        clip_values=(0.0, 1.0),
    )

    transform = create_jatic_dataset_transform(model.preprocessor)

    return classifier, transform


def load_torchvision_model():
    print("Loading torchvision model from jatic_toolbox")
    model = track_params(load_jatic_model)(
        provider="torchvision",
        model_name="resnet18",
        task="image-classification",
    )

    classifier = track_init_params(PyTorchClassifier)(
        JaticImageClassificationModel(model),
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
        input_shape=(224, 224, 3),
        channels_first=False,
        nb_classes=10,
        clip_values=(0.0, 1.0),
    )

    transform = create_jatic_dataset_transform(model.preprocessor)

    return classifier, transform


def main():
    parser = create_parser(
        description="Run example using models and datasets from the JATIC toolbox",
        batch_size=16,
        export_every_n_batches=5,
        num_batches=5,
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
        train_dataset, test_dataset = load_torchvision_dataset(
            transform, args.batch_size
        )
    else:
        train_dataset, test_dataset = load_huggingface_dataset(
            transform, args.batch_size
        )

    dataset = Dataset(
        name="CIFAR10",
        x_key="image",
        y_key="label",
        train_dataloader=train_dataset,
        test_dataloader=test_dataset,
    )

    model = Model(
        name="BEiT-finetuned",
        model=loaded_model,
    )

    ###
    # The rest of this file was directly copied from the existing cifar example
    ###

    attack = Attack(
        name="PGD",
        attack=track_init_params(art.attacks.evasion.ProjectedGradientDescent)(
            loaded_model,
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
    )

    evaluation = Evaluation(
        name="CIFAR10-classification",
        description="Baseline cifar10 image classification",
        author="msw@example.com",
        dataset=dataset,
        model=model,
        attack=attack,
        metric=metric,
    )

    task = ImageClassificationTask(
        evaluation, num_classes=10, export_every_n_batches=args.export_every_n_batches
    )

    engine = EvaluationEngine(task, limit_test_batches=args.num_batches)
    results = engine.run()
    print_outputs(dataset, model, results)

    print("JATIC Experiment Complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
