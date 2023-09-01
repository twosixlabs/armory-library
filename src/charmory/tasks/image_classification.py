"""Image classification evaluation task"""

import torch
import torchmetrics.classification

from charmory.metrics.perturbation import PerturbationNormMetric
from charmory.tasks.base import BaseEvaluationTask


class ImageClassificationTask(BaseEvaluationTask):
    """Image classification evaluation task"""

    def __init__(
        self,
        *args,
        num_classes: int,
        perturbation_ord: float = torch.inf,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.benign_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.attack_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.perturbation = PerturbationNormMetric(ord=perturbation_ord)

    def export_batch(self, batch: BaseEvaluationTask.Batch):
        self._export("x", batch.x, batch.i)
        if batch.x_adv is not None:
            self._export("x_adv", batch.x_adv, batch.i)

    def _export(self, name, batch_data, batch_idx):
        batch_size = batch_data.shape[0]
        for sample_idx in range(batch_size):
            filename = f"batch_{batch_idx}_ex_{sample_idx}_{name}.png"
            self.exporter.log_image(batch_data[sample_idx], filename)

    def run_benign(self, batch: BaseEvaluationTask.Batch):
        super().run_benign(batch)
        self.benign_accuracy(torch.tensor(batch.y_pred), torch.tensor(batch.y))
        self.log("benign_accuracy", self.benign_accuracy)

    def run_attack(self, batch: BaseEvaluationTask.Batch):
        super().run_attack(batch)
        self.attack_accuracy(torch.tensor(batch.y_pred_adv), torch.tensor(batch.y))
        self.log("attack_accuracy", self.attack_accuracy)

        self.perturbation(torch.tensor(batch.x), torch.tensor(batch.x_adv))
        self.log("perturbation", self.perturbation)
