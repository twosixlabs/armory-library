"""
Example programmatic entrypoint for scenario execution
"""
import json
from pprint import pprint
import sys

import art.attacks.evasion

import armory.baseline_models.pytorch.cifar
import armory.data.datasets
import armory.scenarios.image_classification
import armory.version
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


def main(argv: list = sys.argv[1:]):
    if len(argv) > 0:
        if "--version" in argv:
            print(armory.version.__version__)
            sys.exit(0)

    print("Armory: Example Programmatic Entrypoint for Scenario Execution")

    dataset = Dataset(
        name="CIFAR10",
        test_dataset=armory.data.datasets.cifar10(
            split="test",
            epochs=1,
            batch_size=64,
            shuffle_files=False,
        ),
    )

    model = Model(
        name="cifar",
        model=armory.baseline_models.pytorch.cifar.get_art_model(
            model_kwargs={},
            wrapper_kwargs={},
            weights_path=None,
        ),
    )

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
        export_batches=True,
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

    print("Training model...")
    model.model.fit_generator(
        armory.data.datasets.cifar10(
            split="train",
            epochs=20,
            batch_size=64,
            shuffle_files=True,
        ),
        nb_epochs=20,
    )

    cifar_engine = Engine(baseline)
    results = cifar_engine.run()

    print("=" * 64)
    # print(json.dumps(baseline.asdict(), indent=4, sort_keys=True))
    # Have altered the json formatted printing in favor for a pprint as the new Evaluation objects contain nonseriabalizable fields which create issues
    pprint(baseline)
    print("-" * 64)
    print(
        json.dumps(
            results, default=lambda o: "<not serializable>", indent=4, sort_keys=True
        )
    )

    print("=" * 64)
    print(dataset)
    print("-" * 64)
    print(model)

    print("=" * 64)
    print("CIFAR10 Experiment Complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
