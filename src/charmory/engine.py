from typing import Optional, Union

import lightning.pytorch as pl

from armory.logs import log
from charmory.evaluation import Evaluation
from charmory.tasks.base import BaseEvaluationTask
from charmory.track import lightning_logger, track_evaluation, track_metrics


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


class LightningEngine(pl.Trainer):
    def __init__(
        self,
        limit_test_batches: Optional[Union[int, float]] = None,
        num_nodes: int = 1,
        strategy: str = "auto",
    ):
        super().__init__(
            inference_mode=False,
            limit_test_batches=limit_test_batches,
            logger=lightning_logger(),
            num_nodes=num_nodes,
            strategy=strategy,
        )
        self.run_id: Optional[str] = None

    def test(self, task: BaseEvaluationTask):
        if self.run_id:
            raise RuntimeError(
                "Evaluation engine has already been run. Create a new LightningEngine "
                "instance to perform a subsequent run."
            )

        active_run = track_evaluation(
            name=task.evaluation.name, description=task.evaluation.description
        )
        self.run_id = active_run.info.run_id
        return super().test(task, dataloaders=task.evaluation.dataset.test_dataset)
        # return dict(
        #     compute=self.task.evaluation.metric.profiler.results(),
        #     metrics=self.trainer.callback_metrics,
        # )
