from contextlib import nullcontext
from typing import Any, Dict, Optional

import mlflow

from armory.logs import log
from charmory.evaluation import Evaluation


class Engine:
    def __init__(self, evaluation: Evaluation, enable_tracking=False):
        self.evaluation = evaluation
        self.enable_tracking = enable_tracking
        self.scenario = evaluation.scenario.function(self.evaluation)

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

    def _track(self):
        """Create context manager for tracking the evaluation run"""
        if not self.enable_tracking:
            return nullcontext()

        else:
            experiment = mlflow.get_experiment_by_name(self.evaluation.name)
            if experiment:
                experiment_id = experiment.experiment_id
            else:
                experiment_id = mlflow.create_experiment(self.evaluation.name)

            return mlflow.start_run(
                experiment_id=experiment_id,
                description=self.evaluation.description,
                tags={
                    "author": self.evaluation.author,
                },
            )

    def _track_results(
        self,
        active_run: Optional[mlflow.ActiveRun],
        results: Dict[str, Any],
        force_track: bool,
    ):
        """Record evaluation run results"""
        if not self.enable_tracking and not force_track:
            return

        # assert active_run

        for key, values in results["results"].items():
            if key == "compute":
                continue
            for val in values:
                mlflow.log_metric(key, val)

    def run(self, track=False):
        with self._track() as active_run:
            results = self.scenario.evaluate()
            self._track_results(active_run, results, track)

        return results
