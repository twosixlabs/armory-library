"""
Base Armory scenario Lightning module
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Optional

import lightning.pytorch as pl
import torch

from armory import metrics
from armory.instrument import Probe
from armory.instrument.export import ExportMeter, PredictionMeter
from charmory.evaluation import Evaluation


class BaseScenario(pl.LightningModule, ABC):
    """Base Armory scenario"""

    def __init__(
        self,
        evaluation: Evaluation,
        export_batches: bool = False,
        skip_benign: bool = False,
        skip_attack: bool = False,
        skip_misclassified: bool = False,
    ):
        super().__init__()
        self.evaluation = evaluation
        self.export_batches = export_batches
        self.skip_benign = skip_benign
        self.skip_attack = skip_attack
        self.skip_misclassified = skip_misclassified
        self.probe = Probe("scenario", evaluation.metric.logger.hub)

    ###
    # Inner classes
    ###

    @dataclass
    class Batch:
        """Iteration batch being evaluated during scenario evaluation"""

        i: int
        x: Any
        y: Any
        y_pred: Optional[Any] = None
        y_target: Optional[Any] = None
        x_adv: Optional[Any] = None
        y_pred_adv: Optional[Any] = None
        misclassified: Optional[bool] = None

    ###
    # Methods required to be implemented by scenario-specific subclasses
    ###

    @abstractmethod
    def _load_sample_exporter(self, export_dir: str):
        ...

    ###
    # Internal methods
    ###

    def _load_export_meters(self, num_export_batches: int, sample_exporter, export_dir):
        for probe_value in ["x", "x_adv"]:
            export_meter = ExportMeter(
                f"{probe_value}_exporter",
                sample_exporter,
                f"scenario.{probe_value}",
                max_batches=num_export_batches,
            )
            self.evaluation.metric.logger.hub.connect_meter(
                export_meter, use_default_writers=False
            )
            if self.skip_attack:
                break

        pred_meter = PredictionMeter(
            "pred_dict_exporter",
            export_dir,
            y_probe="scenario.y",
            y_pred_clean_probe="scenario.y_pred" if not self.skip_benign else None,
            y_pred_adv_probe="scenario.y_pred_adv" if not self.skip_attack else None,
            max_batches=num_export_batches,
        )
        self.evaluation.metric.logger.hub.connect_meter(
            pred_meter, use_default_writers=False
        )

    def _run_benign(self, batch: Batch):
        self.evaluation.metric.logger.hub.set_context(stage="benign")

        batch.x.flags.writeable = False
        with self.evaluation.metric.profiler.measure("Inference"):
            batch.y_pred = self.evaluation.model.model.predict(
                batch.x, **self.evaluation.model.predict_kwargs
            )
        self.probe.update(y_pred=batch.y_pred)

        if self.skip_misclassified:
            batch.misclassified = not any(
                metrics.task.batch.categorical_accuracy(batch.y, batch.y_pred)
            )

    @torch.enable_grad()
    def _run_attack(self, batch: Batch):
        if TYPE_CHECKING:
            assert self.evaluation.attack

        self.evaluation.metric.logger.hub.set_context(stage="attack")
        x = batch.x
        y = batch.y
        y_pred = batch.y_pred

        with self.evaluation.metric.profiler.measure("Attack"):
            # Don't generate the attack if the benign was already misclassified
            if self.skip_misclassified and batch.misclassified:
                y_target = None
                x_adv = x

            else:
                # If targeted, use the label targeter to generate the target label
                if self.evaluation.attack.targeted:
                    if TYPE_CHECKING:
                        assert self.evaluation.attack.label_targeter
                    y_target = self.evaluation.attack.label_targeter.generate(y)
                else:
                    # If untargeted, use either the natural or benign labels
                    # (when set to None, the ART attack handles the benign label)
                    y_target = (
                        y if self.evaluation.attack.use_label_for_untargeted else None
                    )

                x_adv = self.evaluation.attack.attack.generate(
                    x=x, y=y_target, **self.evaluation.attack.generate_kwargs
                )

        self.evaluation.metric.logger.hub.set_context(stage="adversarial")
        # Don't evaluate the attack if the benign was already misclassified
        if self.skip_misclassified and batch.misclassified:
            y_pred_adv = y_pred
        else:
            # Ensure that input sample isn't overwritten by model
            x_adv.flags.writeable = False
            y_pred_adv = self.evaluation.model.model.predict(
                x_adv, **self.evaluation.model.predict_kwargs
            )

        self.probe.update(x_adv=x_adv, y_pred_adv=y_pred_adv)
        if self.evaluation.attack.targeted:
            self.probe.update(y_target=y_target)

        batch.x_adv = x_adv
        batch.y_target = y_target
        batch.y_pred_adv = y_pred_adv

    ###
    # LightningModule method overrides
    ###

    def on_test_start(self):
        self.time_stamp = time.time()
        self.evaluation_id = str(self.time_stamp)

        num_export_batches = 0
        sample_exporter = None
        export_dir = None
        if self.export_batches:
            num_export_batches = len(self.evaluation.dataset.test_dataset)
            export_dir = f"{self.evaluation.sysconfig.paths['output_dir']}/{self.evaluation_id}/outputs"
            Path(export_dir).mkdir(parents=True, exist_ok=True)
            sample_exporter = self._load_sample_exporter(export_dir)

        self._load_export_meters(num_export_batches, sample_exporter, export_dir)

    def on_test_end(self):
        self.metric_results = self.evaluation.metric.logger.results()
        self.compute_results = self.evaluation.metric.profiler.results()
        self.results = {}
        self.results.update(self.metric_results)
        self.results["compute"] = self.compute_results
        self.evaluation.metric.logger.hub.set_context(stage="finished")

    def test_dataloader(self):
        return self.evaluation.dataset.test_dataset

    def test_step(self, batch, batch_idx):
        self.evaluation.metric.logger.hub.set_context(stage="test_step")
        x, y = batch
        self.evaluation.metric.logger.hub.set_context(batch=batch_idx)
        self.probe.update(i=batch_idx, x=x, y=y)
        curr_batch = self.Batch(i=batch_idx, x=x, y=y)
        if not self.skip_benign:
            self._run_benign(curr_batch)
        if not self.skip_attack:
            self._run_attack(curr_batch)
