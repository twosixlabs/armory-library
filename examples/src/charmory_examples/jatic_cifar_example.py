"""
Example programmatic entrypoint for scenario execution
"""
import json
from pprint import pprint
import sys

import art.attacks.evasion

import armory.baseline_models.pytorch.cifar
import armory.scenarios.image_classification
import armory.version
from armory.data.datasets import cifar10_canonical_preprocessing, cifar10_context
from charmory.datasets import (
    JaticVisionDatasetGenerator,
    JaticVisionDataloaderGenerator,
)
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
from jatic_toolbox._internals.interop.evaluate import get_dataloader
from jatic_toolbox import (
    load_dataset as load_jatic_dataset,
    __version__ as jatic_version,
)


def load_huggingface_dataset(
    split: str, epochs: int, batch_size: int, shuffle_files: bool, **kwargs
):
    print(
        "Loading HuggingFace dataset from jatic_toolbox, "
        f"{split=}, {batch_size=}, {epochs=}, {shuffle_files=}"
    )
    dataset = load_jatic_dataset(
        provider="huggingface",
        dataset_name="cifar10",
        task="image-classification",
        split=split,
    )
    return JaticVisionDatasetGenerator(
        dataset=dataset,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=shuffle_files,
        preprocessing_fn=cifar10_canonical_preprocessing,
        context=cifar10_context,
    )


def load_torchvision_dataset(
    split: str, epochs: int, batch_size: int, shuffle_files: bool, **kwargs
):
    print(
        "Loading torchvision dataset from jatic_toolbox, "
        f"{split=}, {batch_size=}, {epochs=}, {shuffle_files=}"
    )
    dataset = load_jatic_dataset(
        provider="torchvision",
        dataset_name="CIFAR10",
        task="image-classification",
        split=split,
        root="/tmp/torchvision_datasets",
        download=True,
    )
    return JaticVisionDatasetGenerator(
        dataset=dataset,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=shuffle_files,
        preprocessing_fn=cifar10_canonical_preprocessing,
        context=cifar10_context,
    )


def load_dataloader(
    split: str, epochs: int, batch_size: int, shuffle_files: bool, **kwargs
):
    print(
        "Loading HuggingFace dataset and dataloader from jatic_toolbox, "
        f"{split=}, {batch_size=}, {epochs=}, {shuffle_files=}"
    )
    # works with either provider
    dataset = load_jatic_dataset(
        provider="huggingface",
        dataset_name="cifar10",
        task="image-classification",
        split=split,
    )
    # dataset = load_jatic_dataset(
    #     provider="torchvision",
    #     dataset_name="CIFAR10",
    #     task="image-classification",
    #     split=split,
    #     root="/tmp/torchvision_datasets",
    #     download=True,
    # )
    dataloader = get_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        split=split,
        shuffle=shuffle_files,
    )
    return JaticVisionDataloaderGenerator(
        dataloader=dataloader,
        size=len(dataset),
        batch_size=batch_size,
        epochs=epochs,
        preprocessing_fn=cifar10_canonical_preprocessing,
        context=cifar10_context,
    )


def main(argv: list = sys.argv[1:]):
    if len(argv) > 0:
        if "--version" in argv:
            print(f"armory: {armory.version.__version__}")
            print(f"JATIC-toolbox: {jatic_version}")
            sys.exit(0)

    load_dataset = load_huggingface_dataset
    if "torchvision" in argv:
        load_dataset = load_torchvision_dataset
    if "dataloader" in argv:
        load_dataset = load_dataloader

    print("Armory: Example Programmatic Entrypoint for Scenario Execution")

    dataset = Dataset(
        function=load_dataset,
        framework="numpy",
        batch_size=64,
    )

    model = Model(
        function=armory.baseline_models.pytorch.cifar.get_art_model,
        model_kwargs={},
        wrapper_kwargs={},
        weights_file=None,
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
        name="cifar_baseline",
        description="Baseline cifar10 image classification",
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

    cifar_engine = Engine(baseline)
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
    print(cifar_engine.dataset)
    print("-" * 64)
    print(cifar_engine.model)

    print("=" * 64)
    print("CIFAR10 Experiment Complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
