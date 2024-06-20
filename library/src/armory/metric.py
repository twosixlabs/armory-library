from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional

import jsonpath_ng
import torch
import torch.nn as nn
from typing_extensions import Self

from armory.data import Batch, DataSpecification, TorchSpec
from armory.track import Trackable

if TYPE_CHECKING:
    from torchmetrics import Metric as TorchMetric


class Metric(Trackable, nn.Module, ABC):
    """
    Base class for an Armory-compatible metric.
    """

    def __init__(
        self,
        metric: "TorchMetric",
        spec: Optional[DataSpecification] = None,
        record_as_artifact: bool = True,
        record_as_metrics: Optional[Iterable[str]] = None,
    ):
        """
        Initializes the metric.

        Args:
            metric: torchmetrics metric to be wrapped.
            spec: Optional, data specification for the batch fields used for the
                metric. This may be used for input data or for predictions from
                the batch. By default, a generic torch spec is used.
            record_as_artifact: If True, the metric result will be recorded as
                an artifact to the evaluation run.
            record_as_metrics: Optional, a set of JSON paths in the metric
                result pointing to scalar values to record as metrics to the
                evaluation run. If None, no metrics will be recorded.
        """
        super().__init__()
        self.metric = metric
        self.spec = spec or TorchSpec()
        self.record_as_artifact = record_as_artifact
        self.record_as_metrics = (
            {path: jsonpath_ng.parse(path) for path in record_as_metrics}
            if record_as_metrics is not None
            else None
        )

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

    @classmethod
    def to_json(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: cls.to_json(v) for k, v in value.items()}

        if isinstance(value, torch.Tensor):
            value = value.to(torch.float32)
            if value.dim() == 0:
                return value.item()
            elif value.dim() == 1:
                return [v.item() for v in value]
            else:
                return [cls.to_json(v) for v in value]

        return float(value)

    def get_scalars(self, value: Any) -> Dict[str, float]:
        if not self.record_as_metrics or value is None:
            return {}

        scalars = {}
        for path, expr in self.record_as_metrics.items():
            matches = expr.find(value)
            if not matches:
                raise RuntimeError(f"{path} did not match any values in metric result")
            if len(matches) > 1:
                raise RuntimeError(
                    f"{path} matched multiple values in metric result, only one value allowed"
                )
            scalar = matches[0].value
            if not isinstance(scalar, float):
                raise RuntimeError(
                    f"{path} matched a non-scalar value in metric result: {scalar}"
                )
            scalars[path] = scalar

        return scalars

    def clone(self) -> Self:
        """Creates a clone of the metric."""
        return self.__class__(
            metric=self.metric,
            spec=self.spec,
            record_as_artifact=self.record_as_artifact,
            record_as_metrics=self.record_as_metrics,
        )

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

    def update(self, batch: Batch) -> None:
        if batch.predictions is not None:
            self.metric.update(
                batch.predictions.get(self.spec),
                batch.targets.get(self.spec),
            )
