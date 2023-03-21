"""A mock-up demo to show how Armory and MLflow interact."""

from charmory.evaluation import Evaluation
from charmory.blocks import mnist

from loguru import logger as log
import mlflow


class Evaluator:
    def __init__(self, eval: Evaluation):
        self.eval = eval

        metadata = eval._metadata
        mlexp = mlflow.get_experiment_by_name(metadata.name)
        if mlexp:
            self.experiment_id = mlexp.experiment_id
            log.info(f"Experiment {metadata.name} already exists {self.experiment_id}")
        else:
            self.experiment_id = mlflow.create_experiment(metadata.name)
            log.info(f"Creating experiment {metadata.name} as {self.experiment_id}")

    def run(self):
        """fake an evaluation to demonstrate mlflow tracking."""
        metadata = self.eval._metadata
        log.info("Starting mlflow run:")
        self.show()
        self.active_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            description=metadata.description,
            tags={
                "author": self.eval._metadata.author,
            },
        )

        # fake variable epsilon and results
        import random

        epsilon = random.random()
        result = {"benign": epsilon, "adversarial": 1 - epsilon}
        assert self.eval.attack
        self.eval.attack.kwargs["eps"] = epsilon

        for key, value in self.eval.flatten():
            if key.startswith("_metadata."):
                continue
            mlflow.log_param(key, value)

        for k, v in result.items():
            mlflow.log_metric(k, v)

        mlflow.end_run()
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
