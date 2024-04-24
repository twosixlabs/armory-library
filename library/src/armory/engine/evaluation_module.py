"""
Armory lightning module to perform evaluations
"""

from typing import Any, Mapping

import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
import torch

from armory.data import Batch
from armory.evaluation import Chain
from armory.export.sink import MlflowSink, Sink
from armory.metrics.compute import Profiler


class EvaluationModule(pl.LightningModule):
    """
    Armory lightning module to perform an evaluation on a single evaluation chain
    """

    def __init__(
        self,
        chain: Chain,
        profiler: Profiler,
    ):
        """
        Initializes the lightning module.

        Args:
            chain: Evaluation chain
        """
        super().__init__()
        self.chain = chain
        self.profiler = profiler
        # store model as an attribute so it gets moved to device automatically
        assert chain.model is not None
        self.model = chain.model

        # Make copies of user-configured metrics for the chain
        self.metrics = self.MetricsDict(
            {
                metric_name: metric.clone()
                for metric_name, metric in chain.metrics.items()
            }
        )

    ###
    # Inner classes
    ###

    class MetricsDict(torch.nn.ModuleDict):
        def update_metrics(self, batch: "Batch") -> None:
            for metric in self.values():
                metric.update(batch)

        def compute(self) -> Mapping[str, torch.Tensor]:
            return {name: metric.compute() for name, metric in self.items()}

        def reset(self) -> None:
            for metric in self.values():
                metric.reset()

    ###
    # Task evaluation methods
    ###

    def apply_perturbations(self, batch: "Batch"):
        """
        Applies the chain's perturbations to the batch to produce the perturbed data
        to be given to the model
        """
        with self.profiler.measure(f"{self.chain.name}/perturbation"):
            for perturbation in self.chain.perturbations:
                with self.profiler.measure(
                    f"{self.chain.name}/perturbation/{perturbation.name}"
                ):
                    perturbation.apply(batch)

    def evaluate(self, batch: "Batch"):
        """Perform evaluation on batch"""
        with self.profiler.measure(f"{self.chain.name}/predict"):
            self.model.predict(batch)

    def update_metrics(self, batch: "Batch"):
        self.metrics.update_metrics(batch)

    def log_metric(self, name: str, metric: Any):
        if isinstance(metric, dict):
            for k, v in metric.items():
                self.log_metric(f"{name}/{k}", v)

        elif isinstance(metric, torch.Tensor):
            metric = metric.to(torch.float32)
            if len(metric.shape) == 0:
                self.log(name, metric)
            elif len(metric.shape) == 1:
                self.log_dict(
                    {f"{name}/{idx}": value for idx, value in enumerate(metric)},
                    sync_dist=True,
                )
            else:
                for idx, value in enumerate(metric):
                    self.log_metric(f"{name}/{idx}", value)

        else:
            self.log(name, metric)

    ###
    # LightningModule method overrides
    ###

    def setup(self, stage: str) -> None:
        """Sets up the exporters"""
        super().setup(stage)
        logger = self.logger
        sink = (
            MlflowSink(logger.experiment, logger.run_id)
            if isinstance(logger, MLFlowLogger)
            else Sink()
        )
        for exporter in self.chain.exporters:
            exporter.use_sink(sink)

    def on_test_epoch_start(self) -> None:
        """Resets all metrics"""
        self.metrics.reset()
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        """
        Performs evaluations of the model for each configured perturbation chain
        """
        try:
            with torch.enable_grad():
                self.apply_perturbations(batch)
            self.evaluate(batch)
            self.update_metrics(batch)

            for exporter in self.chain.exporters:
                exporter.export(batch_idx, batch)
        except BaseException as err:
            raise RuntimeError(
                f"Error performing evaluation of batch #{batch_idx} in chain '{self.chain.name}': {batch}"
            ) from err

    def on_test_epoch_end(self) -> None:
        """Logs all metric results"""
        for metric_name, metric in self.metrics.items():
            self.log_metric(f"{metric_name}", metric.compute())

        return super().on_test_epoch_end()
