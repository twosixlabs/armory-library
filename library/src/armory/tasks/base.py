"""
Base Armory evaluation task
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Mapping, Optional

import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
import numpy as np
import torch

from armory.evaluation import Evaluation
from armory.export import Exporter, MlflowExporter

if TYPE_CHECKING:
    from armory.perturbation import Perturbation

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

        # Make copies of user-configured metrics for each perturbation chain
        self.perturbation_metrics = self.MetricsDict(
            {
                chain_name: self.MetricsDict(
                    {
                        metric_name: metric.clone()
                        for metric_name, metric in self.evaluation.metric.perturbation.items()
                    }
                )
                for chain_name in self.evaluation.perturbations.keys()
            }
        )
        self.prediction_metrics = self.MetricsDict(
            {
                chain_name: self.MetricsDict(
                    {
                        metric_name: metric.clone()
                        for metric_name, metric in self.evaluation.metric.prediction.items()
                    }
                )
                for chain_name in self.evaluation.perturbations.keys()
            }
        )

    ###
    # Inner classes
    ###

    @dataclass
    class Batch:
        """Batch being evaluated during each step of the evaluation task"""

        chain_name: str
        i: int
        data: Mapping[str, Any]
        x_key: str
        y_key: str
        perturbation_output: Dict[str, Any] = field(default_factory=dict)
        x_perturbed: Optional[Any] = None
        y_predicted: Optional[Any] = None

        @property
        def x(self):
            return self.data[self.x_key]

        @property
        def y(self):
            return self.data[self.y_key]

    class MetricsDict(torch.nn.ModuleDict):
        def update_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> None:
            for metric in self.values():
                metric.update(pred, target)

        def compute(self) -> Mapping[str, torch.Tensor]:
            return {name: metric.compute() for name, metric in self.items()}

        def reset(self) -> None:
            for metric in self.values():
                metric.reset()

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

    def create_batch(self, chain_name, batch, batch_idx):
        """Creates a batch object from the given dataset-specific batch"""
        return self.Batch(
            chain_name=chain_name,
            i=batch_idx,
            data=batch,
            x_key=self.evaluation.dataset.x_key,
            y_key=self.evaluation.dataset.y_key,
        )

    def apply_perturbations(self, batch: Batch, chain: Iterable["Perturbation"]):
        """
        Applies the given perturbation chain to the batch to produce the perturbed data
        to be given to the model
        """
        x = batch.x
        with self.evaluation.metric.profiler.measure(
            f"{batch.chain_name}/perturbation"
        ):
            for perturbation in chain:
                with self.evaluation.metric.profiler.measure(
                    f"{batch.chain_name}/perturbation/{perturbation.name}"
                ):
                    x, out = perturbation.apply(x, batch)
                    batch.perturbation_output[perturbation.name] = out
        batch.x_perturbed = x

    def evaluate(self, batch: Batch):
        """Perform evaluation on batch"""
        assert isinstance(batch.x_perturbed, np.ndarray)
        # Ensure that input sample isn't overwritten by model
        batch.x_perturbed.flags.writeable = False
        with self.evaluation.metric.profiler.measure(f"{batch.chain_name}/predict"):
            batch.y_predicted = self.evaluation.model.model.predict(
                batch.x_perturbed, **self.evaluation.model.predict_kwargs
            )

    def update_metrics(self, batch: Batch):
        x = torch.tensor(batch.x).to(self.device)
        x_perturbed = torch.tensor(batch.x_perturbed).to(self.device)
        y = self.target_to_tensor(batch.y)
        y_predicted = self.target_to_tensor(batch.y_predicted)

        self.perturbation_metrics[batch.chain_name].update_metrics(x, x_perturbed)
        self.prediction_metrics[batch.chain_name].update_metrics(y_predicted, y)

    def target_to_tensor(self, target):
        return torch.tensor(target).to(self.device)

    def log_metric(self, name: str, metric: Any):
        if isinstance(metric, dict):
            for k, v in metric.items():
                self.log_metric(f"{name}/{k}", v)

        elif isinstance(metric, torch.Tensor):
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

    @staticmethod
    def _from_list(maybe_list, idx):
        try:
            return maybe_list[idx]
        except:  # noqa: E722
            # if it's None or is not a list/sequence/etc, just return None
            return None

    def export_batch_metadata(self, batch: Batch):
        data_keys = set(batch.data.keys()) - {
            self.evaluation.dataset.x_key,
            self.evaluation.dataset.y_key,
        }
        for sample_idx in range(batch.x.shape[0]):
            dictionary = dict(
                y=batch.y[sample_idx],
                y_predicted=self._from_list(batch.y_predicted, sample_idx),
            )
            for key in data_keys:
                dictionary[key] = self._from_list(batch.data[key], sample_idx)
            for name, output in batch.perturbation_output.items():
                if isinstance(output, Mapping):
                    dictionary.update(
                        {k: self._from_list(v, sample_idx) for k, v in output.items()}
                    )
                else:
                    dictionary[name] = self._from_list(output, sample_idx)
            self.exporter.log_dict(
                dictionary=dictionary,
                artifact_file=f"batch_{batch.i}_ex_{sample_idx}_{batch.chain_name}.txt",
            )

    ###
    # LightningModule method overrides
    ###

    def on_test_epoch_start(self) -> None:
        """Resets all metrics"""
        self.perturbation_metrics.reset()
        self.prediction_metrics.reset()
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        """
        Performs evaluations of the model for each configured perturbation chain
        """
        for name, chain in self.evaluation.perturbations.items():
            chain_batch = self.create_batch(name, batch, batch_idx)

            with torch.enable_grad():
                self.apply_perturbations(chain_batch, chain)
            self.evaluate(chain_batch)
            self.update_metrics(chain_batch)

            if self._should_export(batch_idx):
                self.export_batch(chain_batch)

    def on_test_epoch_end(self) -> None:
        """Logs all metric results"""
        for chain_name, chain in self.perturbation_metrics.items():
            for metric_name, metric in chain.items():
                self.log_metric(f"{chain_name}/{metric_name}", metric.compute())

        for chain_name, chain in self.prediction_metrics.items():
            for metric_name, metric in chain.items():
                self.log_metric(f"{chain_name}/{metric_name}", metric.compute())

        return super().on_test_epoch_end()
