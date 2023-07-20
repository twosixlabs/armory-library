"""
Example programmatic entrypoint for scenario execution
"""
import json
from pprint import pprint
import sys

import art.attacks.evasion
from jatic_toolbox import __version__ as jatic_version
from jatic_toolbox import load_dataset as load_jatic_dataset

import armory.baseline_models.pytorch.pokemon
from armory.data.datasets import pokemon_context, pokemon_preprocessing
import armory.scenarios.image_classification
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

# Loads Pokemon Classification HuggingFace Example
def load_huggingface_dataset(
    split: str, epochs: int, batch_size: int, shuffle_files: bool, **kwargs
):
    print(
        "Loading HuggingFace dataset from jatic_toolbox, "
        f"{split=}, {batch_size=}, {epochs=}, {shuffle_files=}"
    )
    dataset = load_jatic_dataset(
        provider="huggingface",
        dataset_name="keremberke/pokemon-classification",
        task="image-classification",
        name='full',
        split=split,
    )
    return JaticVisionDatasetGenerator(
        dataset=dataset,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=shuffle_files,
        preprocessing_fn=pokemon_preprocessing,
        context=pokemon_context,
    )




def main(argv: list = sys.argv[1:]):
    if len(argv) > 0:
        if "--version" in argv:
            print(f"armory: {armory.version.__version__}")
            print(f"JATIC-toolbox: {jatic_version}")
            sys.exit(0)



    print("Armory: Example Programmatic Entrypoint for Scenario Execution")

    dataset = Dataset(
        name="POKEMON",
        train_dataset=load_huggingface_dataset(
            split="train",
            epochs=20,
            batch_size=64,
            shuffle_files=True,
        ),
        test_dataset=load_huggingface_dataset(
            split="test",
            epochs=1,
            batch_size=64,
            shuffle_files=False,
        ),
    )

    pokemon_model = armory.baseline_models.pytorch.pokemon.get_art_model(
        model_kwargs={},
        wrapper_kwargs={},
    )

    model = Model(
        name="pokemon",
        model=pokemon_model,
        fit=True,
        fit_kwargs={"nb_epochs": 20},
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
        function=armory.scenarios.image_classification.ImageClassificationTask,
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
