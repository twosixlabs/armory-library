"""Image classification evaluation task"""

import numpy as np
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
        self._export_image("x", batch.x, batch.i)
        if batch.x_adv is not None:
            self._export_image("x_adv", batch.x_adv, batch.i)
        self._export_targets(batch)

    def _export_image(self, name, batch_data, batch_idx):
        batch_size = batch_data.shape[0]
        for sample_idx in range(batch_size):
            filename = f"batch_{batch_idx}_ex_{sample_idx}_{name}.png"
            self.exporter.log_image(batch_data[sample_idx], filename)

    @staticmethod
    def _serialize(numpy_array_or_list):
        if isinstance(numpy_array_or_list, np.ndarray):
            return numpy_array_or_list.tolist()
        return numpy_array_or_list

    def _export_targets(self, batch):
        targets = dict(
            i=batch.i,
            y=self._serialize(batch.y),
            y_pred=self._serialize(batch.y_pred),
            y_target=self._serialize(batch.y_target),
            y_pred_adv=self._serialize(batch.y_pred_adv),
        )
        self.exporter.log_dict(
            dictionary=targets,
            artifact_file=f"batch_{batch.i}_targets.txt",
        )

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
