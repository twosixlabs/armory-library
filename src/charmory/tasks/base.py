"""
Base Armory evaluation task
"""

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import lightning.pytorch as pl
import torch

from charmory.evaluation import Evaluation


class BaseEvaluationTask(ABC):
    """Base Armory evaluation task"""

    def __init__(
        self,
        evaluation: Evaluation,
        skip_benign: bool = False,
        skip_attack: bool = False,
    ):
        self.evaluation = evaluation
        self.skip_benign = skip_benign
        self.skip_attack = skip_attack

    ###
    # Inner classes
    ###

    @dataclass
    class Batch:
        """Batch being evaluated during each step of the evaluation task"""

        i: int
        x: Any
        y: Any
        y_pred: Optional[Any] = None
        y_target: Optional[Any] = None
        x_adv: Optional[Any] = None
        y_pred_adv: Optional[Any] = None

    ###
    # Methods required to be implemented by task-specific subclasses
    ###

    ###
    # Optional methods to be implemented by task-specific subclasses
    ###

    ###
    # Task execution methods
    ###

    def on_test_start(self):
        pass

    def run_benign(self, batch: Batch):
        """Perform benign evaluation"""
        # Ensure that input sample isn't overwritten by model
        batch.x.flags.writeable = False
        batch.y_pred = self.evaluation.model.model.predict(
            batch.x, **self.evaluation.model.predict_kwargs
        )

    def run_attack(self, batch: Batch):
        """Perform adversarial evaluation"""
        if TYPE_CHECKING:
            assert self.evaluation.attack

        # If targeted, use the label targeter to generate the target label
        if self.evaluation.attack.targeted:
            if TYPE_CHECKING:
                assert self.evaluation.attack.label_targeter
            batch.y_target = self.evaluation.attack.label_targeter.generate(batch.y)
        else:
            # If untargeted, use either the natural or benign labels
            # (when set to None, the ART attack handles the benign label)
            batch.y_target = (
                batch.y if self.evaluation.attack.use_label_for_untargeted else None
            )

        batch.x_adv = self.evaluation.attack.attack.generate(
            x=batch.x, y=batch.y_target, **self.evaluation.attack.generate_kwargs
        )

        # Ensure that input sample isn't overwritten by model
        batch.x_adv.flags.writeable = False
        batch.y_pred_adv = self.evaluation.model.model.predict(
            batch.x_adv, **self.evaluation.model.predict_kwargs
        )

    def on_test_end(self):
        pass


class EvaluationTaskModule(pl.LightningModule):
    """Lightning module to execute an Armory evaluation task"""

    def __init__(self, task: BaseEvaluationTask):
        super().__init__()
        self.task = task

    def on_test_start(self):
        self.task.on_test_start()

    def on_test_end(self):
        self.task.on_test_end()

    def test_dataloader(self):
        return self.task.evaluation.dataset.test_dataset

    def test_step(self, batch, batch_idx):
        """Invokes task's benign and adversarial evaluations"""
        x, y = batch
        curr_batch = self.task.Batch(i=batch_idx, x=x, y=y)
        if not self.task.skip_benign:
            self.task.run_benign(curr_batch)
        if not self.task.skip_attack:
            with torch.enable_grad():
                self.task.run_attack(curr_batch)
