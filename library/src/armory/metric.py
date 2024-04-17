from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch.nn as nn

from armory.data import Batch, DataSpecification, TorchSpec

if TYPE_CHECKING:
    from torchmetrics import Metric as TorchMetric


class Metric(nn.Module, ABC):
    """
    Base class for an Armory-compatible metric.
    """

    def __init__(self, metric: "TorchMetric", spec: Optional[DataSpecification] = None):
        """
        Initializes the metric.

        Args:
            metric: torchmetrics metric to be wrapped.
            spec: Optional, data specification for the batch fields used for the
                metric. This may be used for input data or for predictions from
                the batch. By default, a generic torch spec is used.
        """
        super().__init__()
        self.metric = metric
        self.spec = spec or TorchSpec(device=self.metric.device)

    def _apply(self, *args, **kwargs):
        super()._apply(*args, **kwargs)
        if isinstance(self.spec, TorchSpec):
            self.spec.to(device=self.metric.device)

    def compute(self):
        """Computes the metric value(s)."""
        return self.metric.compute()

    def reset(self):
        """Resets the metric."""
        self.metric.reset()

    @abstractmethod
    def clone(self) -> "Metric":
        """Creates a clone of the metric."""
        ...

    @abstractmethod
    def update(self, batch: Batch) -> None:
        """Updates the metric with a batch from the evaluation."""
        ...


class PerturbationMetric(Metric):
    """
    A metric based on comparing the unperturbed input data against the final
    perturbed data used as input to the model.

    The wrapped metric's `update` method must accept two values:

    - The unperturbed input data
    - The final perturbed input data

    Example::

        from armory.metric import PerturbationMetric
        from armory.metrics.perturbation import PerturbationNormMetric

        metric = PerturbationMetric(PerturbationNormMetric())
    """

    def clone(self):
        return PerturbationMetric(self.metric.clone(), self.spec)

    def update(self, batch: Batch) -> None:
        self.metric.update(
            batch.initial_inputs.get(self.spec),
            batch.inputs.get(self.spec),
        )


class PredictionMetric(Metric):
    """
    A metric based on comparing the natural, or ground truth, targets against
    the model's predictions.

    The wrapped metric's `update` method must accept two values:

    - The model's predictions
    - The ground truth targets

    Example::

        from torchmetrics.classification import Accuracy
        from armory.metric import PredictionMetric

        metric = PredictionMetric(Accuracy())
    """

    def clone(self):
        return PredictionMetric(self.metric.clone(), self.spec)

    def update(self, batch: Batch) -> None:
        if batch.predictions is not None:
            self.metric.update(
                batch.predictions.get(self.spec),
                batch.targets.get(self.spec),
            )
