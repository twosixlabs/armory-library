"""
Example programmatic entrypoint for scenario execution
"""

import sys

import art.attacks.evasion
from jatic_toolbox import __version__ as jatic_version
from jatic_toolbox import load_dataset as load_jatic_dataset

import armory.baseline_models.pytorch.pokemon
from armory.metrics.compute import BasicProfiler
import armory.version
from charmory.data import ArmoryDataLoader
from charmory.engine import EvaluationEngine
from charmory.evaluation import Attack, Dataset, Evaluation, Metric, Model
from charmory.experimental.example_results import print_outputs
from charmory.experimental.transforms import create_image_classification_transform
from charmory.tasks.image_classification import ImageClassificationTask
from charmory.track import track_init_params, track_params

BATCH_SIZE = 16


# Loads Pokemon Classification HuggingFace Example


def load_huggingface_dataset():
    transform = create_image_classification_transform(
        image_from_np=lambda img: img,
    )

    train_dataset = track_params(load_jatic_dataset)(
        provider="huggingface",
        dataset_name="keremberke/pokemon-classification",
        task="image-classification",
        name="full",
        split="train",
    )

    train_dataset.set_transform(transform)

    train_dataloader = ArmoryDataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_dataset = track_params(load_jatic_dataset)(
        provider="huggingface",
        dataset_name="keremberke/pokemon-classification",
        task="image-classification",
        name="full",
        split="test",
    )
    test_dataset.set_transform(transform)
    test_dataloader = ArmoryDataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
    )
    return train_dataloader, test_dataloader


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
        name="POKEMON",
        x_key="image",
        y_key="label",
        train_dataloader=train_dataset,
        test_dataloader=test_dataset,
    )

    ###
    # The rest of this file was directly copied from the existing cifar example
    ###

    attack = Attack(
        name="PGD",
        attack=track_init_params(art.attacks.evasion.ProjectedGradientDescent)(
            pokemon_model,
            batch_size=3,
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
        name="pokemon",
        description="Baseline Pokemon image classification",
        author="msw@example.com",
        dataset=dataset,
        model=model,
        attack=attack,
        metric=metric,
    )

    task = ImageClassificationTask(
        evaluation, num_classes=150, export_every_n_batches=5
    )
    engine = EvaluationEngine(task, limit_test_batches=5)
    results = engine.run()

    print_outputs(dataset, model, results)

    print("Pokemon Experiment Complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
