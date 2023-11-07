"""Perturbation metrics"""

import torch
from torchmetrics import Metric


class PerturbationNormMetric(Metric):
    """Metric for L-norm distance between ground truth and perturbed samples"""

    def __init__(self, ord: float = torch.inf):
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
