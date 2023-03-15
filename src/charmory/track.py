"""A mock-up demo to show how Armory and MLflow interact."""

from mlflow import log_metric, log_param

from armory.logs import log
from charmory.experiment import (
    Attack,
    Dataset,
    Experiment,
    MetaData,
    Metric,
    Model,
    Scenario,
    SysConfig,
)


class Evaluator:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment

    def run(self):
        log.info(f"Running experiment {self.experiment._metadata.name}")
        result = {}
        result["benign"] = id(self.experiment.model)
        if self.experiment.attack:
            result["attack"] = id(self.experiment.attack)
        return result


log.info("Starting demo")


def mnist_experiment() -> Experiment:
    return Experiment(
        _metadata=MetaData(
            name="mnist experiment",
            description="derived from mnist_baseline.json",
            author="msw@example.com",
        ),
        model=Model(
            function="armory.baseline_models.keras.mnist.get_art_model",
            model_kwargs={},
            wrapper_kwargs={},
            weights_file=None,
            fit=True,
            fit_kwargs={"nb_epochs": 20},
        ),
        scenario=Scenario(
            function="armory.scenarios.image_classification.ImageClassificationTask",
            kwargs={},
        ),
        dataset=Dataset(
            function="armory.data.datasets.mnist", framework="numpy", batch_size=128
        ),
        attack=Attack(
            function="art.attacks.evasion.FastGradientMethod",
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


def main():
    experiment = mnist_experiment()
    evaluator = Evaluator(experiment)
    exp_dict = experiment.asdict()
    for block in exp_dict:
        log_param(block, exp_dict[block])
    log.success("mnist experiment tracked")

    results = evaluator.run()
    for item in results:
        log_metric(item, results[item])
    log.success("mnist experiment results tracked")


if __name__ == "__main__":
    main()
