from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Self

import torch.nn as nn

from charmory.batch import DefaultTorchAccessor, TorchAccessor

if TYPE_CHECKING:
    from torchmetrics import Metric as TorchMetric

    from charmory.batch import Accessor, Batch


class Metric(nn.Module, ABC):
    def __init__(self, metric: "TorchMetric", accessor: Optional["Accessor"] = None):
        super().__init__()
        self.metric = metric
        self.accessor = accessor or DefaultTorchAccessor(device=self.metric.device)

    def _apply(self, *args, **kwargs):
        super()._apply(*args, **kwargs)
        if isinstance(self.accessor, TorchAccessor):
            self.accessor.to(device=self.metric.device)

    def compute(self):
        return self.metric.compute()

    def reset(self):
        self.metric.reset()

    @abstractmethod
    def clone(self) -> Self:
        ...

    @abstractmethod
    def update(self, batch: "Batch") -> None:
        ...


class PerturbationMetric(Metric):
    def clone(self):
        return PerturbationMetric(self.metric.clone(), self.accessor)

    def update(self, batch: "Batch") -> None:
        self.metric.update(
            self.accessor.get(batch.initial_inputs),
            self.accessor.get(batch.inputs),
        )


class PredictionMetric(Metric):
    def clone(self):
        return PredictionMetric(self.metric.clone(), self.accessor)

    def update(self, batch: "Batch") -> None:
        if batch.predictions is not None:
            self.metric.update(
                self.accessor.get(batch.predictions),
                self.accessor.get(batch.targets),
            )
