from typing import Optional, Union

import lightning.pytorch as pl
import lightning.pytorch.loggers as pl_loggers
from lightning.pytorch.utilities import rank_zero_only

from charmory.tasks.base import BaseEvaluationTask
from charmory.track import get_current_params, init_tracking_uri


class EvaluationEngine:
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
                "Evaluation engine has already been run. Create a new EvaluationEngine "
                "instance to perform a subsequent run."
            )
        self._was_run = True

        self._log_params()
        self.trainer.test(
            self.task, dataloaders=self.task.evaluation.dataset.test_dataloader
        )
        return dict(
            compute=self.task.evaluation.metric.profiler.results(),
            metrics=self.trainer.callback_metrics,
        )
