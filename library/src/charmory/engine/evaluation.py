"""Armory engine to perform model robustness evaluations"""
from typing import Mapping, Optional, TypedDict

import lightning.pytorch as pl
import lightning.pytorch.loggers as pl_loggers
from lightning.pytorch.utilities import rank_zero_only
from torch import Tensor

from charmory.tasks.base import BaseEvaluationTask
from charmory.track import get_current_params, init_tracking_uri, track_system_metrics


class EvaluationResults(TypedDict):
    """Robustness evaluation results"""

    compute: Mapping[str, float]
    """Computational metrics"""
    metrics: Mapping[str, Tensor]
    """Task-specific evaluation metrics"""


class EvaluationEngine:
    """
    Armory engine to perform model robustness evaluations.

    Example::

        from charmory.engine import EvaluationEngine

        # assuming `task` has been defined using a `charmory.tasks` class
        engine = EvaluationEngine(task)
        results = engine.run()
    """

    def __init__(
        self, task: BaseEvaluationTask, run_id: Optional[str] = None, **kwargs
    ):
        """
        Initializes the engine.

        Args:
            task: Armory evaluation task to perform model inference and
                application of adversarial attacks
            run_id: Optional, MLflow run ID to which to record evaluation results
            **kwargs: All other keyword arguments will be forwarded to the
                `lightning.pytorch.Trainer` class.
        """
        self.task = task
        self._logger = pl_loggers.MLFlowLogger(
            experiment_name=self.task.evaluation.name,
            tags={"mlflow.note.content": self.task.evaluation.description},
            tracking_uri=init_tracking_uri(self.task.evaluation.sysconfig.armory_home),
            run_id=run_id,
        )
        self.trainer = pl.Trainer(
            inference_mode=False,
            logger=self._logger,
            **kwargs,
        )
        self.run_id = run_id
        self._was_run = False

    @rank_zero_only
    def _log_params(self):
        """Log tracked params with MLflow"""
        self.run_id = self._logger.run_id
        self._logger.log_hyperparams(get_current_params())

    def run(self) -> EvaluationResults:
        """Perform the evaluation"""
        if self._was_run:
            raise RuntimeError(
                "Evaluation engine has already been run. Create a new EvaluationEngine "
                "instance to perform a subsequent run."
            )
        self._was_run = True

        self._log_params()
        assert self.run_id, "No run ID was created by the MLflow logger"
        with track_system_metrics(self.run_id):
            self.trainer.test(
                self.task, dataloaders=self.task.evaluation.dataset.test_dataloader
            )
        return EvaluationResults(
            compute=self.task.evaluation.metric.profiler.results(),
            metrics=self.trainer.callback_metrics,
        )
