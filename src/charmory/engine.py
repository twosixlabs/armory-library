from typing import Any, Dict, Optional

import mlflow

from armory.logs import log
from charmory.evaluation import Evaluation
from charmory.track import track_evaluation, track_metrics


class Engine:
    def __init__(self, evaluation: Evaluation):
        self.evaluation = evaluation
        self.scenario = evaluation.scenario.function(self.evaluation)
        self._run_id: Optional[str] = None

    def train(self, nb_epochs=1):
        """
        Train the evaluation model using the configured training dataset.

        Args:
            nb_epochs: Number of epochs with which to perform training
        """
        assert self.evaluation.dataset.train_dataset is not None, (
            "Requested to train the model but the evaluation dataset does not "
            "provide a train_dataset"
        )
        log.info(
            f"Fitting {self.evaluation.model.name} model with "
            f"{self.evaluation.dataset.name} dataset..."
        )
        # TODO trainer defense when poisoning attacks are supported
        self.evaluation.model.model.fit_generator(
            self.evaluation.dataset.train_dataset,
            nb_epochs=nb_epochs,
        )

    def run(self):
        if self._run_id:
            raise RuntimeError(
                "Evaluation engine has already been run. Create a new Engine "
                "instance to perform a subsequent run."
            )

        with track_evaluation(
            name=self.evaluation.name, description=self.evaluation.description
        ) as active_run:
            self._run_id = active_run.info.run_id
            results = self.scenario.evaluate()
            track_metrics(results["results"]["metrics"])

            return results

    def log_params(self, params: Dict[str, Any]):
        if not self._run_id:
            raise RuntimeError("Engine must be run before `log_params` may be called.")
        with mlflow.start_run(run_id=self._run_id):
            mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if not self._run_id:
            raise RuntimeError("Engine must be run before `log_metric` may be called.")
        with mlflow.start_run(run_id=self._run_id):
            mlflow.log_metrics(metrics, step)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        if not self._run_id:
            raise RuntimeError(
                "Engine must be run before `log_artifacts` may be called."
            )
        with mlflow.start_run(run_id=self._run_id):
            mlflow.log_artifacts(local_dir, artifact_path)
