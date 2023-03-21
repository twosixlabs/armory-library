"""convenient pre-fabricated "canned" armory evaluation experiments"""

from charmory.evaluation import (
    Attack,
    Dataset,
    Evaluation,
    MetaData,
    Metric,
    Model,
    Scenario,
    SysConfig,
)


def mnist_baseline() -> Evaluation:
    return Evaluation(
        _metadata=MetaData(
            name="mnist_baseline",
            description="derived from mnist_baseline.json",
            author="msw@example.com",
        ),
        model=Model(
            function="armory.baseline_models.keras.mnist:get_art_model",
            model_kwargs={},
            wrapper_kwargs={},
            weights_file=None,
            fit=True,
            fit_kwargs={"nb_epochs": 20},
        ),
        scenario=Scenario(
            function="armory.scenarios.image_classification:ImageClassificationTask",
            kwargs={},
        ),
        dataset=Dataset(
            function="armory.data.datasets:mnist", framework="numpy", batch_size=128
        ),
        attack=Attack(
            function="art.attacks.evasion:FastGradientMethod",
            kwargs={
                "batch_size": 1,
                "eps": 0.2,
                "eps_step": 0.1,
                "minimal": False,
                "num_random_init": 0,
                "targeted": False,
            },
            knowledge="white",
            use_label=True,
            type=None,
        ),
        defense=None,
        metric=Metric(
            profiler_type="basic",
            supported_metrics=["accuracy"],
            perturbation=["linf"],
            task=["categorical_accuracy"],
            means=True,
            record_metric_per_sample=False,
        ),
        sysconfig=SysConfig(gpus=["all"], use_gpu=True),
    )
