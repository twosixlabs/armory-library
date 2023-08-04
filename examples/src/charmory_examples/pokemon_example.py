"""
Example programmatic entrypoint for scenario execution
"""
import json
from pprint import pprint
import sys

import art.attacks.evasion
from jatic_toolbox import __version__ as jatic_version
from jatic_toolbox import load_dataset as load_jatic_dataset
import torch

import armory.baseline_models.pytorch.pokemon
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
from charmory.utils import PILtoNumpy_HuggingFace

BATCH_SIZE = 16
TRAINING_EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Loads Pokemon Classification HuggingFace Example


def load_huggingface_dataset():
    transform = PILtoNumpy_HuggingFace()

    train_dataset = load_jatic_dataset(
        provider="huggingface",
        dataset_name="keremberke/pokemon-classification",
        task="image-classification",
        name="full",
        split="train",
    )

    train_dataset.set_transform(transform)

    train_dataset_generator = JaticVisionDatasetGenerator(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        epochs=TRAINING_EPOCHS,
        shuffle=True,
    )
    test_dataset = load_jatic_dataset(
        provider="huggingface",
        dataset_name="keremberke/pokemon-classification",
        task="image-classification",
        name="full",
        split="test",
    )
    test_dataset.set_transform(transform)
    test_dataset_generator = JaticVisionDatasetGenerator(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        epochs=1,
    )
    return train_dataset_generator, test_dataset_generator


def main(argv: list = sys.argv[1:]):
    if len(argv) > 0:
        if "--version" in argv:
            print(f"armory: {armory.version.__version__}")
            print(f"JATIC-toolbox: {jatic_version}")
            sys.exit(0)

    print("Armory: Example Programmatic Entrypoint for Scenario Execution")

    pokemon_model = armory.baseline_models.pytorch.pokemon.get_art_model(
        model_kwargs={},
        wrapper_kwargs={},
        weights_path=None,
    )

    model = Model(
        name="pokemon",
        model=pokemon_model,
    )

    train_dataset, test_dataset = load_huggingface_dataset()
    dataset = Dataset(
        name="POKEMON", train_dataset=train_dataset, test_dataset=test_dataset
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
        name="pokemon",
        description="Baseline Pokemon image classification",
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

    pokemon_engine = Engine(baseline)
    pokemon_engine.train(nb_epochs=TRAINING_EPOCHS)
    results = pokemon_engine.run()

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
    print("Pokemon Experiment Complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
