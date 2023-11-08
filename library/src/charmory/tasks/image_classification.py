"""Image classification evaluation task"""

from typing import Union

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
        perturbation_ord: Union[float, int] = torch.inf,
        **kwargs,
    ):
        """
        Initializes the task.

        Args:
            *args: All positional arguments will be forwarded to the
                `charmory.tasks.base.BaseEvaluationTask` class
            num_classes: Total number of classes the model is capable of
                predicting, used for categorical accuracy metrics
            perturbation_ord: L-norm order for the perturbation distance metrics
            **kwargs: All other keyword arguments will be forwarded to the
                `charmory.tasks.base.BaseEvaluationTask` class
        """
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
            image = batch_data[sample_idx]
            if self.export_adapter is not None:
                image = self.export_adapter(image)
            self.exporter.log_image(image, filename)

    @staticmethod
    def _from_list(maybe_list, idx):
        try:
            return maybe_list[idx]
        except:  # noqa: E722
            # if it's None or is not a list/sequence/etc, just return None
            return None

    def _export_targets(self, batch: BaseEvaluationTask.Batch):
        keys = set(batch.data.keys()) - {
            self.evaluation.dataset.x_key,
            self.evaluation.dataset.y_key,
        }
        for sample_idx in range(batch.x.shape[0]):
            dictionary = dict(
                y=batch.y[sample_idx],
                y_pred=self._from_list(batch.y_pred, sample_idx),
                y_target=self._from_list(batch.y_target, sample_idx),
                y_pred_adv=self._from_list(batch.y_pred_adv, sample_idx),
            )
            for k in keys:
                dictionary[k] = self._from_list(batch.data[k], sample_idx)
            self.exporter.log_dict(
                dictionary=dictionary,
                artifact_file=f"batch_{batch.i}_ex_{sample_idx}_y.txt",
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
