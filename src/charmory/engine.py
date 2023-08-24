from typing import Optional

from armory.logs import log
from charmory.evaluation import Evaluation
from charmory.track import track_evaluation, track_metrics


class Engine:
    def __init__(self, evaluation: Evaluation):
        self.evaluation = evaluation
        self.scenario = evaluation.scenario.function(self.evaluation)
        self.run_id: Optional[str] = None

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
        if self.run_id:
            raise RuntimeError(
                "Evaluation engine has already been run. Create a new Engine "
                "instance to perform a subsequent run."
            )

        with track_evaluation(
            name=self.evaluation.name, description=self.evaluation.description
        ) as active_run:
            self.run_id = active_run.info.run_id
            results = self.scenario.evaluate()
            track_metrics(results["results"]["metrics"])

            return results
