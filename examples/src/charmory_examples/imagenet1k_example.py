"""
Example programmatic entrypoint for scenario execution
"""
import json
from pprint import pprint
import sys

import art.attacks.evasion
from jatic_toolbox import __version__ as jatic_version
from jatic_toolbox import load_dataset as load_jatic_dataset

from armory.baseline_models.pytorch.imagenet1k import get_art_model
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
from charmory.track import track_init_params
from charmory.utils import PILtoNumpy_HuggingFace_Variable_Length

BATCH_SIZE = 16
TRAINING_EPOCHS = 1


# Loads Imagenet 1k Classification HuggingFace Example


def load_huggingface_dataset():
    transform = PILtoNumpy_HuggingFace_Variable_Length()
    train_dataset = load_jatic_dataset(
        provider="huggingface",
        dataset_name="imagenet-1k",
        task="image-classification",
        split="train",
        use_auth_token=True,
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
        dataset_name="imagenet-1k",
        task="image-classification",
        split="test",
        use_auth_token=True,
    )
    test_dataset.set_transform(transform)
    test_dataset_generator = JaticVisionDatasetGenerator(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        epochs=1,
        size=512,  # Use a subset just for demo purposes
    )
    return train_dataset_generator, test_dataset_generator


def main(argv: list = sys.argv[1:]):
    if len(argv) > 0:
        if "--version" in argv:
            print(f"armory: {armory.version.__version__}")
            print(f"JATIC-toolbox: {jatic_version}")
            sys.exit(0)

    print("Armory: Example Programmatic Entrypoint for Scenario Execution")

    image_net_model = get_art_model(
        model_kwargs={},
        wrapper_kwargs={},
        weights_path=None,
    )

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
        name="imagenet1k",
        description="Baseline imagenet1k image classification",
        author="msw@example.com",
        dataset=dataset,
        model=model,
        attack=attack,
        scenario=scenario,
        metric=metric,
        sysconfig=sysconfig,
    )

    print(f"Starting Demo for {baseline.name}")

    image_net_engine = Engine(baseline)
    image_net_engine.train(nb_epochs=TRAINING_EPOCHS)
    results = image_net_engine.run()

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
    print("Imagenet 1k Experiment Complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
