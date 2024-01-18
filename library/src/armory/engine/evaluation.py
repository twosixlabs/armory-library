"""Armory engine to perform model robustness evaluations"""
from typing import Mapping, Optional, TypedDict

import lightning.pytorch as pl
import lightning.pytorch.loggers as pl_loggers
from lightning.pytorch.utilities import rank_zero_only
from torch import Tensor

from armory.engine.evaluation_module import EvaluationModule
from armory.evaluation import Evaluation
from armory.track import (
    get_current_params,
    init_tracking_uri,
    track_param,
    track_system_metrics,
)
import armory.version


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

        # assuming `evaluation` has been defined using `charmory.evaluation.Evaluation`
        engine = EvaluationEngine(evaluation)
        results = engine.run()
    """

    def __init__(
        self,
        evaluation: Evaluation,
        export_every_n_batches: int = 0,
        run_id: Optional[str] = None,
        **kwargs,
    ):
        """
        Initializes the engine.

        Args:
            evaluation: Configuration for the evaluation
            export_every_n_batches: Frequency at which batches will be exported
                to MLflow. A value of 0 means that no batches will be exported.
                The data that is exported is task-specific.
            run_id: Optional, MLflow run ID to which to record evaluation results
            **kwargs: All other keyword arguments will be forwarded to the
                `lightning.pytorch.Trainer` class.
        """
        self.evaluation = evaluation
        self._logger = pl_loggers.MLFlowLogger(
            experiment_name=evaluation.name,
            tags={"mlflow.note.content": evaluation.description},
            tracking_uri=init_tracking_uri(evaluation.sysconfig.armory_home),
            run_id=run_id,
        )
        self.module = EvaluationModule(evaluation, export_every_n_batches)
        self.trainer = pl.Trainer(
            inference_mode=False,
            logger=self._logger,
            **kwargs,
        )
        self.run_id = run_id
        self._was_run = False

    @property
    def metrics(self) -> EvaluationModule.MetricsDict:
        """
        The dictionary mapping perturbation chain names to a dictionary mapping
        metric names to the metric objects.
        """
        return self.module.metrics

    @rank_zero_only
    def _log_params(self):
        """Log tracked params with MLflow"""
        track_param("Armory.version", armory.version.__version__)
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
                self.module, dataloaders=self.evaluation.dataset.dataloader
            )
        return EvaluationResults(
            compute=self.evaluation.profiler.results(),
            metrics=self.trainer.callback_metrics,
        )
