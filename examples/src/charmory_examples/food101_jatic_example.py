import json
from pprint import pprint
import sys

import art.attacks.evasion
from jatic_toolbox import load_dataset as load_jatic_dataset
import torch
from torchvision import transforms as T

import armory.baseline_models.pytorch.food101
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
from charmory.utils import PILtoNumpy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
TRAINING_EPOCHS = 1
ROOT = "/home/rahul/cache"


def load_torchvision_dataset(root_path):
    print("Loading torchvision dataset from jatic_toolbox")
    train_dataset = load_jatic_dataset(
        provider="torchvision",
        dataset_name="Food101",
        task="image-classification",
        split="train",
        root=root_path,
        download=True,
        transform=T.Compose(
            [
                T.Resize(size=(512, 512)),
                PILtoNumpy(),
            ]
        ),
    )
    train_dataset_generator = JaticVisionDatasetGenerator(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        epochs=TRAINING_EPOCHS,
        shuffle=True,
    )
    test_dataset = load_jatic_dataset(
        provider="torchvision",
        dataset_name="Food101",
        task="image-classification",
        split="test",
        root=root_path,
        download=True,
        transform=T.Compose(
            [
                T.Resize(size=(512, 512)),
                PILtoNumpy(),
            ]
        ),
    )
    test_dataset_generator = JaticVisionDatasetGenerator(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        epochs=1,
        shuffle=False,
    )

    return train_dataset_generator, test_dataset_generator


def main():
    train_dataset, test_dataset = load_torchvision_dataset(ROOT)

    dataset = Dataset(
        name="Food101",
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )
    model = Model(
        name="Food101",
        model=armory.baseline_models.pytorch.food101.get_art_model(
            model_kwargs={},
            wrapper_kwargs={},
            weights_path=None,
        ),
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
        name="food101_baseline",
        description="Baseline food101 image classification",
        author="msw@example.com",
        dataset=dataset,
        model=model,
        attack=attack,
        scenario=scenario,
        defense=None,
        metric=metric,
        sysconfig=sysconfig,
    )

    print(f"Starting Demo for {baseline.name}")

    food_engine = Engine(baseline)
    food_engine.train(nb_epochs=TRAINING_EPOCHS)
    results = food_engine.run()

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
