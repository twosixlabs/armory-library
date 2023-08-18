"""Image classification evaluation task"""

import torch
import torchmetrics.classification

from armory.instrument.export import ImageClassificationExporter
from charmory.metrics.perturbation import PerturbationNormMetric
from charmory.tasks.base import BaseEvaluationTask, Batch


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

    def create_sample_exporter(self, export_dir):
        return ImageClassificationExporter(export_dir)

    def run_benign(self, batch: Batch):
        super().run_benign(batch)
        self.benign_accuracy(torch.tensor(batch.y_pred), torch.tensor(batch.y))
        self.log("benign_accuracy", self.benign_accuracy)

    def run_attack(self, batch: Batch):
        super().run_attack(batch)
        self.attack_accuracy(torch.tensor(batch.y_pred_adv), torch.tensor(batch.y))
        self.log("attack_accuracy", self.attack_accuracy)

        self.perturbation(torch.tensor(batch.x), torch.tensor(batch.x_adv))
        self.log("perturbation", self.perturbation)
