"""
Example programmatic entrypoint for scenario execution
"""
import json
import sys

import armory.version
from charmory.blocks import cifar10, mnist  # noqa: F401
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
        function="armory.data.datasets:cifar10", framework="numpy", batch_size=64
    )

    model = Model(
        function="armory.baseline_models.pytorch.cifar:get_art_model",
        model_kwargs={},
        wrapper_kwargs={},
        weights_file=None,
        fit=True,
        fit_kwargs={"nb_epochs": 20},
    )

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
