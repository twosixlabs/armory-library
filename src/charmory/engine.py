from typing import Optional, Union

import lightning.pytorch as pl
import lightning.pytorch.loggers as pl_loggers
from lightning.pytorch.utilities import rank_zero_only

from armory.logs import log
from charmory.evaluation import Evaluation
from charmory.tasks.base import BaseEvaluationTask
from charmory.track import (
    get_current_params,
    init_tracking_uri,
    track_evaluation,
    track_metrics,
)


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


class LightningEngine:
    def __init__(
        self,
        task: BaseEvaluationTask,
        limit_test_batches: Optional[Union[int, float]] = None,
    ):
        self.task = task
        self._logger = pl_loggers.MLFlowLogger(
            experiment_name=self.task.evaluation.name,
            tags={"mlflow.note.content": self.task.evaluation.description},
            tracking_uri=init_tracking_uri(self.task.evaluation.sysconfig.armory_home),
        )
        self.trainer = pl.Trainer(
            inference_mode=False,
            limit_test_batches=limit_test_batches,
            logger=self._logger,
        )
        self.run_id: Optional[str] = None
        self._was_run = False

    @rank_zero_only
    def _log_params(self):
        self.run_id = self._logger.run_id
        self._logger.log_hyperparams(get_current_params())

    def run(self):
        if self._was_run:
            raise RuntimeError(
                "Evaluation engine has already been run. Create a new LightningEngine "
                "instance to perform a subsequent run."
            )
        self._was_run = True

        self._log_params()
        self.trainer.test(
            self.task, dataloaders=self.task.evaluation.dataset.test_dataset
        )
        return dict(
            compute=self.task.evaluation.metric.profiler.results(),
            metrics=self.trainer.callback_metrics,
        )
