"""Object detection evaluation task"""

# import torch
# import torchmetrics.classification

from armory.instrument.export import ObjectDetectionExporter

# from charmory.metrics.perturbation import PerturbationNormMetric
from charmory.tasks.base import BaseEvaluationTask


class ObjectDetectionTask(BaseEvaluationTask):
    """Object detection evaluation task"""

    def __init__(
        self,
        *args,
        # num_classes: int,
        # perturbation_ord: float = torch.inf,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # self.benign_accuracy = torchmetrics.classification.Accuracy(
        #     task="multiclass", num_classes=num_classes
        # )
        # self.attack_accuracy = torchmetrics.classification.Accuracy(
        #     task="multiclass", num_classes=num_classes
        # )
        # self.perturbation = PerturbationNormMetric(ord=perturbation_ord)
        self.sample_exporter = ObjectDetectionExporter(self.export_dir)

    ###
    # Export methods
    ###

    def export_batch(self, batch: BaseEvaluationTask.Batch):
        self._export("benign", batch.x, batch.y, batch.y_pred, batch.i)
        if batch.x_adv is not None:
            self._export("attack", batch.x_adv, batch.y, batch.y_pred_adv, batch.i)

    def _export(self, name, images, boxes, preds, batch_idx):
        batch_size = images.shape[0]
        for sample_idx in range(batch_size):
            basename = f"batch_{batch_idx}_ex_{sample_idx}_{name}"
            self.sample_exporter.export(
                x_i=images[sample_idx],
                basename=basename,
                with_boxes=boxes is not None or preds is not None,
                y=boxes[sample_idx] if boxes else None,
                y_pred=preds[sample_idx] if preds else None,
            )

    def run_benign(self, batch: BaseEvaluationTask.Batch):
        super().run_benign(batch)
        # self.benign_accuracy(torch.tensor(batch.y_pred), torch.tensor(batch.y))
        # self.log("benign_accuracy", self.benign_accuracy)

    def run_attack(self, batch: BaseEvaluationTask.Batch):
        super().run_attack(batch)
        # self.attack_accuracy(torch.tensor(batch.y_pred_adv), torch.tensor(batch.y))
        # self.log("attack_accuracy", self.attack_accuracy)

        # self.perturbation(torch.tensor(batch.x), torch.tensor(batch.x_adv))
        # self.log("perturbation", self.perturbation)
