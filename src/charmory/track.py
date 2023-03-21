"""A mock-up demo to show how Armory and MLflow interact."""


from loguru import logger as log
import mlflow

# import charmory.canned


def show_mlflow_experiement(experiment_id):
    experiment = mlflow.get_experiment(experiment_id)
    print(f"Experiment: {experiment.name}")
    print(f"tags: {experiment.tags}")
    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Lifecycle Stage: {experiment.lifecycle_stage}")
    print(f"Creation Time: {experiment.creation_time}")

    def run(self):
        """fake an evaluation to demonstrate mlflow tracking."""
        metadata = self.evaluation._metadata
        log.info("Starting mlflow run:")
        show_mlflow_experiement(self.experiment_id)
        self.active_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            description=metadata.description,
            tags={
                "author": self.evaluation._metadata.author,
            },
        )

        # fake variable epsilon and results
        import random

        epsilon = random.random()
        result = {"benign": epsilon, "adversarial": 1 - epsilon}
        self.evaluation.attack.kwargs["eps"] = epsilon

        for key, value in self.evaluation.flatten():
            if key.startswith("_metadata."):
                continue
            mlflow.log_param(key, value)

        for k, v in result.items():
            mlflow.log_metric(k, v)

        mlflow.end_run()
        return result


def main():
    # mnist = charmory.canned.mnist_baseline()

    # evaluator = Engine(mnist)
    # log.info("mnist experiment tracked")

    # results = evaluator.run()
    # log.info(f"mnist experiment results tracked {results}")
    ...


if __name__ == "__main__":
    main()
