"""
Example programmatic entrypoint for scenario execution
"""

import sys

import art.attacks.evasion
from jatic_toolbox import __version__ as jatic_version
from jatic_toolbox import load_dataset as load_jatic_dataset

import armory.baseline_models.pytorch.pokemon
from armory.instrument.config import MetricsLogger
from armory.metrics.compute import BasicProfiler
import armory.version
from charmory.data import JaticVisionDatasetGenerator
from charmory.engine import LightningEngine
from charmory.evaluation import Attack, Dataset, Evaluation, Metric, Model, SysConfig
from charmory.experimental.example_results import print_outputs
from charmory.tasks.image_classification import ImageClassificationTask
from charmory.track import track_init_params, track_params
from charmory.utils import PILtoNumpy_HuggingFace

BATCH_SIZE = 16
TRAINING_EPOCHS = 20


# Loads Pokemon Classification HuggingFace Example


def load_huggingface_dataset():
    transform = PILtoNumpy_HuggingFace()

    train_dataset = track_params(load_jatic_dataset)(
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
    test_dataset = track_params(load_jatic_dataset)(
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
        name="pokemon",
        description="Baseline Pokemon image classification",
        author="msw@example.com",
        dataset=dataset,
        model=model,
        attack=attack,
        metric=metric,
        sysconfig=sysconfig,
    )

    task = ImageClassificationTask(
        evaluation, num_classes=150, export_every_n_batches=5
    )
    engine = LightningEngine(task, limit_test_batches=5)
    results = engine.run()

    print_outputs(dataset, model, results)

    print("Pokemon Experiment Complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
