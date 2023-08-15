"""
Base Armory scenario Lightning module
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Optional

import lightning.pytorch as pl

from armory import metrics
from armory.instrument import MetricsLogger, get_hub, get_probe
from armory.instrument.export import ExportMeter, PredictionMeter
from armory.metrics import compute
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

    def _load_metrics(self):
        metrics_config = self.evaluation.metric
        metrics_logger = MetricsLogger.from_config(
            metrics_config,
            include_benign=not self.skip_benign,
            include_adversarial=not self.skip_attack,
            include_targeted=(
                self.evaluation.attack.targeted if self.evaluation.attack else False
            ),
        )
        self.profiler = compute.profiler_from_config(metrics_config)
        self.metrics_logger = metrics_logger

    def _load_export_meters(self, num_export_batches: int, sample_exporter, export_dir):
        for probe_value in ["x", "x_adv"]:
            export_meter = ExportMeter(
                f"{probe_value}_exporter",
                sample_exporter,
                f"scenario.{probe_value}",
                max_batches=num_export_batches,
            )
            self.hub.connect_meter(export_meter, use_default_writers=False)
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
        self.hub.connect_meter(pred_meter, use_default_writers=False)

    def _run_benign(self, batch: Batch):
        self.hub.set_context(stage="benign")

        batch.x.flags.writeable = False
        with self.profiler.measure("Inference"):
            batch.y_pred = self.evaluation.model.model.predict(
                batch.x, **self.evaluation.model.predict_kwargs
            )
        self.probe.update(y_pred=batch.y_pred)

        if self.skip_misclassified:
            batch.misclassified = not any(
                metrics.task.batch.categorical_accuracy(batch.y, batch.y_pred)
            )

    def _run_attack(self, batch: Batch):
        time.sleep(0.15)

    ###
    # LightningModule method overrides
    ###

    def setup(self, stage):
        self.probe = get_probe("scenario")
        self.hub = get_hub()

        self._load_metrics()

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

    def test_dataloader(self):
        return self.evaluation.dataset.test_dataset

    def test_step(self, batch, batch_idx):
        x, y = batch
        curr_batch = self.Batch(i=batch_idx, x=x, y=y)
        # if not self.skip_benign:
        self._run_benign(curr_batch)
        # if not self.skip_attack:
        self._run_attack(curr_batch)
