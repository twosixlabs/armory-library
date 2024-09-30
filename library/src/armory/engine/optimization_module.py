"""
Armory lightning module to perform attack optimizations
"""

import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
import torch

from armory.data import Batch
from armory.evaluation import Optimization
from armory.export.sink import MlflowSink, Sink


class OptimizationModule(pl.LightningModule):
    """
    Armory lightning module to perform attack optimizations
    """
    
    def __init__(self, optimization: Optimization):
        """
        Initializes the lightning module.

        :param optimization: Attack optimization to perform
        :type optimization: Optimization
        """
        super().__init__()
        # store these as attributes so they get moved to device automatically
        self.attack = optimization.attack
        self.transforms = optimization.transforms
        self.model = optimization.model

    def setup(self, stage: str) -> None:
        """
        Sets up the exporters

        :param stage: stage
        :type stage: str
        """
        super().setup(stage)
        logger = self.logger
        self.sink = (
            MlflowSink(logger.experiment, logger.run_id)
            if isinstance(logger, MLFlowLogger)
            else Sink()
        )

    def configure_optimizers(self):
        """
        Configure optimizers
        """
        return self.attack.optimizers()

    def on_train_epoch_end(self):
        """
        On train epoch end
        """
        self.attack.export(self.sink, self.current_epoch)

    def training_step(self, batch: Batch, batch_idx: int):
        """
        Training step

        :param batch: Batch
        :type batch: Batch
        :param batch_idx: Batch index
        :type batch_idx: int
        """
        # Apply the attack to the batch
        self.attack.apply(batch)

        # Apply EOT to the batch
        if self.transforms:
            for transform in self.transforms:
                transform.apply(batch)

        # Calculate the loss
        loss = self.model.loss(batch)
        assert isinstance(loss, torch.Tensor)
        self.log("loss", loss)

        # Return the loss for automatic optimization via Lightning
        return loss
