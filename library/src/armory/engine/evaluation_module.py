"""
Armory lightning module to perform evaluations
"""

from typing import Any, Iterable, Mapping

import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
import torch
import tqdm

from armory.data import Batch
from armory.evaluation import Evaluation, PerturbationProtocol
from armory.export.sink import MlflowSink, Sink


class EvaluationModule(pl.LightningModule):
    """Armory lightning module to perform evaluations"""

    def __init__(
        self,
        evaluation: Evaluation,
        export_every_n_batches: int = 0,
    ):
        """
        Initializes the task.

        Args:
            evaluation: Configuration for the evaluation
            export_every_n_batches: Frequency at which batches will be exported
                to MLflow. A value of 0 means that no batches will be exported.
                The data that is exported is task-specific.
        """
        super().__init__()
        self.evaluation = evaluation
        # store model as an attribute so it gets moved to device automatically
        self.model = evaluation.model
        self.export_every_n_batches = export_every_n_batches

        # Make copies of user-configured metrics for each perturbation chain
        self.metrics = self.MetricsDict(
            {
                chain_name: self.MetricsDict(
                    {
                        metric_name: metric.clone()
                        for metric_name, metric in self.evaluation.metrics.items()
                    }
                )
                for chain_name in self.evaluation.perturbations.keys()
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
    # Task evaluation methods
    ###

    def apply_perturbations(
        self, chain_name: str, batch: "Batch", chain: Iterable["PerturbationProtocol"]
    ):
        """
        Applies the given perturbation chain to the batch to produce the perturbed data
        to be given to the model
        """
        with self.evaluation.profiler.measure(f"{chain_name}/perturbation"):
            for perturbation in chain:
                with self.evaluation.profiler.measure(
                    f"{chain_name}/perturbation/{perturbation.name}"
                ):
                    perturbation.apply(batch)

    def evaluate(self, chain_name: str, batch: "Batch"):
        """Perform evaluation on batch"""
        with self.evaluation.profiler.measure(f"{chain_name}/predict"):
            self.evaluation.model.predict(batch)

    def update_metrics(self, chain_name: str, batch: "Batch"):
        self.metrics[chain_name].update_metrics(batch)

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
        """Sets up the exporter"""
        super().setup(stage)
        logger = self.logger
        sink = (
            MlflowSink(logger.experiment, logger.run_id)
            if isinstance(logger, MLFlowLogger)
            else Sink()
        )
        self.evaluation.exporter.use_sink(sink)

    def on_test_epoch_start(self) -> None:
        """Resets all metrics"""
        self.metrics.reset()
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        """
        Performs evaluations of the model for each configured perturbation chain
        """
        pbar = tqdm.tqdm(self.evaluation.perturbations.items(), position=1, leave=False)
        for chain_name, chain in pbar:
            pbar.set_description(f"Evaluating {chain_name}")

            chain_batch = batch.clone()

            try:
                with torch.enable_grad():
                    self.apply_perturbations(chain_name, chain_batch, chain)
                self.evaluate(chain_name, chain_batch)
                self.update_metrics(chain_name, chain_batch)

                if self._should_export(batch_idx):
                    self.evaluation.exporter.export(chain_name, batch_idx, chain_batch)
            except BaseException as err:
                raise RuntimeError(
                    f"Error performing evaluation of batch #{batch_idx} using chain '{chain_name}': {batch}"
                ) from err

    def on_test_epoch_end(self) -> None:
        """Logs all metric results"""
        for chain_name, chain in self.metrics.items():
            for metric_name, metric in chain.items():
                self.log_metric(f"{chain_name}/{metric_name}", metric.compute())

        return super().on_test_epoch_end()
