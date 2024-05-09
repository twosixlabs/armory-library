from typing import Optional

import lightning
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch

import armory.data
import armory.dataset
import armory.metric
import armory.model


class Trainer(lightning.LightningModule):
    """Base model trainer"""

    def __init__(
        self,
        model: armory.model.ArmoryModel,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        metric: Optional[armory.metric.Metric] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_metric = metric

    def training_step(self, batch: armory.data.Batch, batch_idx: int) -> STEP_OUTPUT:
        inputs = batch.inputs.get(self.model.inputs_spec)
        outputs = self.model(inputs)
        batch.predictions.set(outputs)
        targets = batch.targets.get(armory.data.TorchSpec())
        loss = self.criterion(outputs, targets)
        self.log("train_loss", loss)
        if self.train_metric:
            self.train_metric(batch)
            self.log("foo", self.train_metric.metric)
        return loss

    def validation_step(self, batch: armory.data.Batch, batch_idx: int) -> STEP_OUTPUT:
        inputs = batch.inputs.get(self.model.inputs_spec)
        targets = batch.targets.get(armory.data.TorchSpec())
        predictions = self.model(inputs)
        loss = self.criterion(predictions, targets)
        self.log("val_loss", loss)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.lr_scheduler:
            return [self.optimizer], [self.lr_scheduler]
        else:
            return self.optimizer

    def fit(
        self,
        train_dataloader: armory.dataset.DataLoader,
        val_dataloader: Optional[armory.dataset.DataLoader] = None,
        **kwargs,
    ) -> armory.model.ArmoryModel:
        trainer = lightning.Trainer(
            **kwargs,
        )
        trainer.fit(
            model=self,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        return self.model
