"""
Example programmatic entrypoint for scenario execution
"""
import json
import math
import sys

import armory.version
from armory.data.datasets import ArmoryDataGenerator, cifar10_canonical_preprocessing, cifar10_context
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
from jatic_toolbox import load_dataset as load_jatic_dataset
from jatic_toolbox.protocols import Dataset as JaticDataset, VisionDataset
import numpy as np


class VisionDatasetWrapper:
    """
    Wrapper around JATIC dataset to support iterator interface and return X, Y
    tuple in the correct format
    """

    def __init__(self, dataset: VisionDataset):
        self.dataset = dataset
        self.current = 0
        self.batch_size = 64 # this could be variable/set via argument
        
    def __next__(self):
        stop = min(self.current + self.batch_size, len(self.dataset))
        image = []
        label = []
        for i in range(self.current, stop):
            result = self.dataset[i]
            image.append(np.asarray(result["image"]))
            label.append(result["label"])
        self.current = stop
        if self.current == len(self.dataset):
            # This is weird, have to reset back to beginning of array
            # in order to work with the training epochs. Built-in armory
            # datasets work around this by repeating the data that gets
            # loaded so the final length is dataset-length * num-epochs.
            self.current = 0
        image = np.asarray(image)
        label = np.asarray(label)

        return image, label

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)


def load_dataset(split: str, **kwargs):
    print(f"Loading dataset from jatic_toolbox, {split=}")
    dataset = load_jatic_dataset(
        provider="huggingface",
        dataset_name="cifar10",
        task="image-classification",
        split=split,
    )
    assert isinstance(dataset, JaticDataset)
    return VisionDatasetWrapper(dataset)


def load_dataset_as_adg(split: str, epochs: int, **kwargs):
    print(f"Loading dataset from jatic_toolbox, {split=}, {epochs=}")
    dataset = load_jatic_dataset(
        provider="huggingface",
        dataset_name="cifar10",
        task="image-classification",
        split=split,
    )
    assert isinstance(dataset, JaticDataset)
    return ArmoryDataGenerator(
        generator=VisionDatasetWrapper(dataset),
        size=len(dataset),
        batch_size=64,
        epochs=epochs, 
        preprocessing_fn=cifar10_canonical_preprocessing,
        context=cifar10_context,
    )


def main(argv: list = sys.argv[1:]):
    if len(argv) > 0:
        if "--version" in argv:
            print(armory.version.__version__)
            sys.exit(0)

    print("Armory: Example Programmatic Entrypoint for Scenario Execution")

    dataset = Dataset(
        # To use a JATIC dataset "directly":
        #function="charmory_examples.jatic_hf_example:load_dataset",
        # To use a JATIC dataset wrapped in ArmoryDataGenerator:
        function="charmory_examples.jatic_hf_example:load_dataset_as_adg",

        framework="numpy",
        batch_size=64,
    )

    model = Model(
        function="armory.baseline_models.pytorch.cifar:get_art_model",
        model_kwargs={},
        wrapper_kwargs={},
        weights_file=None,
        # Can't set this to True when not wrapping the dataset in ArmoryDataGenerator
        # because then it isn't an ART DataGenerator subclass (and yields an error).
        fit=True,
        fit_kwargs={"nb_epochs": 20},
    )

    ###
    # The rest of this file was directly copied from the existing cifar example
    ###

    attack = Attack(
        function="art.attacks.evasion:ProjectedGradientDescent",
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
        function="armory.scenarios.image_classification:ImageClassificationTask",
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
    print(json.dumps(baseline.asdict(), indent=4, sort_keys=True))
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
