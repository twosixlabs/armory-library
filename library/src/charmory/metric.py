from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Self

import torch.nn as nn

if TYPE_CHECKING:
    from torchmetrics import Metric as TorchMetric

    from charmory.batch import Accessor, Batch


class Metric(nn.Module, ABC):
    @abstractmethod
    def clone(self) -> Self:
        ...

    def update(self, batch: "Batch") -> None:
        ...

    def compute(self) -> Any:
        ...

    def reset(self) -> None:
        ...


class PerturbationMetric(Metric):
    def __init__(self, metric: "TorchMetric", accessor: "Accessor"):
        super().__init__()
        self.metric = metric
        self.accessor = accessor

    def clone(self):
        return PerturbationMetric(self.metric.clone(), self.accessor)

    def update(self, batch: "Batch") -> None:
        self.metric.update(
            self.accessor.get(batch.initial_inputs), self.accessor.get(batch.inputs)
        )

    def compute(self):
        return self.metric.compute()

    def reset(self):
        self.metric.reset()


class PredictionMetric(Metric):
    def __init__(self, metric: "TorchMetric", accessor: "Accessor"):
        super().__init__()
        self.metric = metric
        self.accessor = accessor

    def clone(self):
        return PredictionMetric(self.metric.clone(), self.accessor)

    def update(self, batch: "Batch") -> None:
        if batch.predictions is not None:
            self.metric.update(
                self.accessor.get(batch.predictions), self.accessor.get(batch.targets)
            )

    def compute(self):
        return self.metric.compute()

    def reset(self):
        self.metric.reset()
