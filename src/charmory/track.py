"""A mock-up demo to show how Armory and MLflow interact."""

from loguru import logger as log
import mlflow

from charmory.blocks import mnist
from charmory.evaluation import Evaluation


class Evaluator:
    def __init__(self, eval: Evaluation):
        self.eval = eval

        mlexp = mlflow.get_experiment_by_name(eval.name)
        if mlexp:
            self.experiment_id = mlexp.experiment_id
            log.info(f"Experiment {eval.name} already exists {self.experiment_id}")
        else:
            self.experiment_id = mlflow.create_experiment(eval.name)
            log.info(f"Creating experiment {eval.name} as {self.experiment_id}")

    def run(self):
        """fake an evaluation to demonstrate mlflow tracking."""
        log.info("Starting mlflow run:")
        self.show()
        self.active_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            description=self.eval.description,
            tags={
                "author": self.eval.author,
            },
        )

        # fake variable epsilon and results
        result = self.fake_result()

        for key, value in self.eval.flatten():
            mlflow.log_param(key, value)

        for k, v in result.items():
            mlflow.log_metric(k, v)

        mlflow.end_run()
        return result

    def fake_result(self):
        import random

        epsilon = random.random()
        result = {"benign": epsilon, "adversarial": 1 - epsilon}
        assert self.eval.attack
        self.eval.attack.kwargs["eps"] = epsilon
        return result

    def show(self):
        experiment = mlflow.get_experiment(experiment_id=self.experiment_id)
        table = {
            "name": experiment.name,
            "tags": experiment.tags,
            "experiment_id": experiment.experiment_id,
            "artifact_location": experiment.artifact_location,
            "lifecycle_stage": experiment.lifecycle_stage,
            "creation_time": experiment.creation_time,
        }
        for label, value in table.items():
            log.info(f"{label}: {value}")


if __name__ == "__main__":
    Evaluator(mnist.baseline).run()
