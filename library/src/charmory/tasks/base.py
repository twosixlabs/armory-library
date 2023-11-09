"""
Base Armory evaluation task
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional

import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
import numpy as np
import torch

from charmory.evaluation import Evaluation
from charmory.export import Exporter, MlflowExporter

ExportAdapter = Callable[[Any], Any]
"""An adapter for exported data (e.g., images). """


class BaseEvaluationTask(pl.LightningModule, ABC):
    """Base Armory evaluation task"""

    def __init__(
        self,
        evaluation: Evaluation,
        skip_benign: bool = False,
        skip_attack: bool = False,
        export_adapter: Optional[ExportAdapter] = None,
        export_every_n_batches: int = 0,
    ):
        """
        Initializes the task.

        Args:
            evaluation: Configuration for the evaluation
            skip_benign: Whether to skip the benign, unperturbed inference
            skip_attack: Whether to skip the adversarial, perturbed inference
            export_adapter: Optional, adapter to be applied to inference data
                prior to exporting to MLflow
            export_every_n_batches: Frequency at which batches will be exported
                to MLflow. A value of 0 means that no batches will be exported.
                The data that is exported is task-specific.
        """
        super().__init__()
        self.evaluation = evaluation
        self.skip_benign = skip_benign
        self.skip_attack = skip_attack
        self.export_adapter = export_adapter
        self.export_every_n_batches = export_every_n_batches
        self._exporter: Optional[Exporter] = None

    ###
    # Inner classes
    ###

    @dataclass
    class Batch:
        """Batch being evaluated during each step of the evaluation task"""

        i: int
        data: Mapping[str, Any]
        x_key: str
        y_key: str
        y_pred: Optional[Any] = None
        y_target: Optional[Any] = None
        x_adv: Optional[Any] = None
        y_pred_adv: Optional[Any] = None

        @property
        def x(self):
            return self.data[self.x_key]

        @property
        def y(self):
            return self.data[self.y_key]

    ###
    # Properties
    ###

    @property
    def exporter(self) -> Exporter:
        """Sample exporter for the current evaluation run"""
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
        """
        Whether the specified batch should be exported, based on the
        `export_every_n_batches` value.
        """
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

    def create_batch(self, batch, batch_idx):
        """Creates a batch object from the given dataset-specific batch"""
        return self.Batch(
            i=batch_idx,
            data=batch,
            x_key=self.evaluation.dataset.x_key,
            y_key=self.evaluation.dataset.y_key,
        )

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
        self.apply_attack(batch)
        assert isinstance(batch.x_adv, np.ndarray)

        # Ensure that input sample isn't overwritten by model
        batch.x_adv.flags.writeable = False
        batch.y_pred_adv = self.evaluation.model.model.predict(
            batch.x_adv, **self.evaluation.model.predict_kwargs
        )

    def apply_attack(self, batch: Batch):
        """Apply attack to batch"""
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

    ###
    # LightningModule method overrides
    ###

    def test_step(self, batch, batch_idx):
        """Invokes task's benign and adversarial evaluations"""
        curr_batch = self.create_batch(batch, batch_idx)
        if not self.skip_benign:
            self.run_benign(curr_batch)
        if not self.skip_attack:
            with torch.enable_grad():
                self.run_attack(curr_batch)
        if self._should_export(batch_idx):
            self.export_batch(curr_batch)
