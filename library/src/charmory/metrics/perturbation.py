"""Perturbation metrics"""

from typing import Union

import torch
from torchmetrics import Metric


class PerturbationNormMetric(Metric):
    """
    Metric for L-norm distance between ground truth and perturbed samples.

    The calculation of the norm depends on the value used for the order of the
    norm, as follows. In the following table, `d` is the distance between ground
    truth and perturbed samples as calculated by `x - x_adv`.

    order        | norm formula
    -------------|-------------
    `torch.inf`  | `max(abs(d))`
    `-torch.inf` | `min(abs(d))`
    0            | `sum(d != 0)`
    1, 2         | `sum(abs(d)**ord)**(1./ord)`
    """

    def __init__(self, ord: Union[float, int] = torch.inf):
        """
        Initializes the perturbation norm distance metric.

        Args:
            ord: order of norm
        """
        super().__init__()
        self.ord = ord
        self.distance: torch.Tensor
        self.total: torch.Tensor
        self.add_state(
            "distance",
            default=torch.tensor(0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, ground: torch.Tensor, perturbed: torch.Tensor):
        self.distance += torch.norm((ground - perturbed).flatten(), p=self.ord)
        self.total += torch.tensor(1)

    def compute(self):
        return self.distance.float() / self.total
