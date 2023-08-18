"""
Base Armory evaluation task
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import TYPE_CHECKING, Any, Optional

import lightning.pytorch as pl
import torch

from armory.instrument.export import SampleExporter
from charmory.evaluation import Evaluation


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


class BatchExporter:
    def __init__(self, export_every_n_batches: int, sample_exporter: SampleExporter):
        self.export_every_n_batches = export_every_n_batches
        self.sample_exporter = sample_exporter

    def should_export(self, batch_idx) -> bool:
        if self.export_every_n_batches == 0:
            return False
        return (batch_idx + 1) % self.export_every_n_batches == 0

    def export_batch(self, batch: Batch):
        if self.should_export(batch.i):
            if batch.x is not None:
                self._export("x", batch.x, batch.i)
            if batch.x_adv is not None:
                self._export("x_adv", batch.x_adv, batch.i)

    def _export(self, name, batch_data, batch_idx):
        print(batch_data.shape)
        batch_size = batch_data.shape[0]
        for sample_idx in range(batch_size):
            basename = f"batch_{batch_idx}_ex_{sample_idx}_{name}"
            self.sample_exporter.export(batch_data[sample_idx], basename)


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
        self.evaluation_id = str(time.time())
        self.export_dir = (
            f"{evaluation.sysconfig.paths['output_dir']}/{self.evaluation_id}"
        )
        self.batch_exporter = BatchExporter(
            export_every_n_batches=export_every_n_batches,
            sample_exporter=self.create_sample_exporter(self.export_dir),
        )

    ###
    # Methods required to be implemented by subclasses
    ###

    @abstractmethod
    def create_sample_exporter(self, export_dir: str) -> SampleExporter:
        """Create task-specific sample exporter"""

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
        curr_batch = Batch(i=batch_idx, x=x, y=y)
        if not self.skip_benign:
            self.run_benign(curr_batch)
        if not self.skip_attack:
            with torch.enable_grad():
                self.run_attack(curr_batch)
        self.batch_exporter.export_batch(curr_batch)
