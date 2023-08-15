"""
Base Armory scenario Lightning module
"""

from abc import ABC  # , abstractmethod
from dataclasses import dataclass
import time
from typing import Any, Optional

import lightning.pytorch as pl

from charmory.evaluation import Evaluation


class BaseScenario(pl.LightningModule, ABC):
    """Base Armory scenario"""

    def __init__(self, evaluation: Evaluation):
        super().__init__()
        self.evaluation = evaluation

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

    ###
    # Internal methods
    ###

    def _run_benign(self, batch: Batch):
        time.sleep(0.1)

    def _run_attack(self, batch: Batch):
        time.sleep(0.15)

    ###
    # LightningModule method overrides
    ###

    def setup(self, stage):
        # load metrics, meters, etc.
        pass

    def test_dataloader(self):
        return self.evaluation.dataset.test_dataset

    def test_step(self, batch, batch_idx):
        x, y = batch
        curr_batch = self.Batch(i=batch_idx, x=x, y=y)
        # if not self.skip_benign:
        self._run_benign(curr_batch)
        # if not self.skip_attack:
        self._run_attack(curr_batch)
