"""
Base Armory evaluation task
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
import torch

from charmory.evaluation import Evaluation
from charmory.export import Exporter, MlflowExporter


class BaseEvaluationTask(pl.LightningModule, ABC):
    """Base Armory evaluation task"""

    def __init__(
        self,
        evaluation: Evaluation,
        skip_benign: bool = False,
        skip_attack: bool = False,
        export_every_n_batches: int = 0,
    ):
        super().__init__()
        self.evaluation = evaluation
        self.skip_benign = skip_benign
        self.skip_attack = skip_attack
        self.export_every_n_batches = export_every_n_batches
        self._exporter: Optional[Exporter] = None

    ###
    # Inner classes
    ###

    @dataclass
    class Batch:
        """Batch being evaluated during each step of the evaluation task"""

        i: int
        x: Any
        y: Any
        y_pred: Optional[Any] = None
        y_target: Optional[Any] = None
        x_adv: Optional[Any] = None
        y_pred_adv: Optional[Any] = None

    ###
    # Properties
    ###

    @property
    def exporter(self) -> Exporter:
        if self._exporter is None:
            logger = self.logger
            if isinstance(logger, MLFlowLogger):
                self._exporter = MlflowExporter(logger.experiment, logger.run_id)
            else:
                self._exporter = Exporter()
        return self._exporter

    ###
    # Internal methods
    ###

    def _should_export(self, batch_idx) -> bool:
        if self.export_every_n_batches == 0:
            return False
        return (batch_idx + 1) % self.export_every_n_batches == 0

    ###
    # Methods required to be implemented by subclasses
    ###

    @abstractmethod
    def export_batch(self, batch: Batch) -> None:
        """Export the given batch"""

    ###
    # Task evaluation methods
    ###

    def run_benign(self, batch: Batch):
        """Perform benign evaluation"""
        # Ensure that input sample isn't overwritten by model
        batch.x.flags.writeable = False
        with self.evaluation.metric.profiler.measure("Inference"):
            batch.y_pred = self.evaluation.model.model.predict(
                batch.x, **self.evaluation.model.predict_kwargs
            )

    def run_attack(self, batch: Batch):
        """Perform adversarial evaluation"""
        if TYPE_CHECKING:
            assert self.evaluation.attack

        with self.evaluation.metric.profiler.measure("Attack"):
            # If targeted, use the label targeter to generate the target label
            if self.evaluation.attack.targeted:
                if TYPE_CHECKING:
                    assert self.evaluation.attack.label_targeter
                batch.y_target = self.evaluation.attack.label_targeter.generate(batch.y)
            else:
                # If untargeted, use either the natural or benign labels
                # (when set to None, the ART attack handles the benign label)
                batch.y_target = (
                    batch.y if self.evaluation.attack.use_label_for_untargeted else None
                )

            batch.x_adv = self.evaluation.attack.attack.generate(
                x=batch.x, y=batch.y_target, **self.evaluation.attack.generate_kwargs
            )

        # Ensure that input sample isn't overwritten by model
        batch.x_adv.flags.writeable = False
        batch.y_pred_adv = self.evaluation.model.model.predict(
            batch.x_adv, **self.evaluation.model.predict_kwargs
        )

    ###
    # LightningModule method overrides
    ###

    def test_step(self, batch, batch_idx):
        """Invokes task's benign and adversarial evaluations"""
        x, y = batch
        curr_batch = self.Batch(i=batch_idx, x=x, y=y)
        if not self.skip_benign:
            self.run_benign(curr_batch)
        if not self.skip_attack:
            with torch.enable_grad():
                self.run_attack(curr_batch)
        if self._should_export(batch_idx):
            self.export_batch(curr_batch)
