"""A mock-up demo to show how Armory and MLflow interact."""

import mlflow

from loguru import logger as log
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


def show_mlflow_experiement(experiment_id):
    experiment = mlflow.get_experiment(experiment_id)
    print(f"Experiment: {experiment.name}")
    print(f"tags: {experiment.tags}")
    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Lifecycle Stage: {experiment.lifecycle_stage}")
    print(f"Creation Time: {experiment.creation_time}")


class Evaluator:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment

        metadata = experiment._metadata
        mlexp = mlflow.get_experiment_by_name(metadata.name)
        if mlexp:
            self.experiment_id = mlexp.experiment_id
            log.info(f"Experiment {metadata.name} already exists {self.experiment_id}")
        else:
            self.experiment_id = mlflow.create_experiment(
                metadata.name,
            )
            log.info(
                f"Creating experiment {self.experiment._metadata.name} as {self.experiment_id}"
            )

    def run(self):
        """fake an evaluation to demonstrate mlflow tracking."""
        metadata = self.experiment._metadata
        log.info("Starting mlflow run:")
        show_mlflow_experiement(self.experiment_id)
        self.active_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            description=metadata.description,
            tags={
                "author": self.experiment._metadata.author,
            },
        )

        # fake variable epsilon and results
        import random

        epsilon = random.random()
        result = {"benign": epsilon, "adversarial": 1 - epsilon}
        self.experiment.attack.kwargs["eps"] = epsilon

        for key, value in self.experiment.flatten():
            if key.startswith("_metadata."):
                continue
            mlflow.log_param(key, value)

        for k, v in result.items():
            mlflow.log_metric(k, v)

        mlflow.end_run()
        return result


def mnist_experiment() -> Experiment:
    return Experiment(
        _metadata=MetaData(
            name="mnist_baseline",
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
    log.info("mnist experiment tracked")

    results = evaluator.run()
    log.info(f"mnist experiment results tracked {results}")


if __name__ == "__main__":
    main()
