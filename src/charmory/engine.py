from typing import Optional, Union

import lightning.pytorch as pl

from charmory.tasks.base import BaseEvaluationTask
from charmory.track import lightning_logger, track_evaluation


class LightningEngine:
    def __init__(
        self,
        task: BaseEvaluationTask,
        limit_test_batches: Optional[Union[int, float]] = None,
    ):
        self.task = task
        self.active_run = track_evaluation(
            name=self.task.evaluation.name, description=self.task.evaluation.description
        )
        self.trainer = pl.Trainer(
            inference_mode=False,
            limit_test_batches=limit_test_batches,
            logger=lightning_logger(),
        )
        self.run_id: Optional[str] = None

    def run(self):
        if self.run_id:
            raise RuntimeError(
                "Evaluation engine has already been run. Create a new LightningEngine "
                "instance to perform a subsequent run."
            )

        self.run_id = self.active_run.info.run_id
        self.trainer.test(
            self.task, dataloaders=self.task.evaluation.dataset.test_dataset
        )
        return dict(
            compute=self.task.evaluation.metric.profiler.results(),
            metrics=self.trainer.callback_metrics,
        )
