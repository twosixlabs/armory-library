"""Image classification evaluation task"""

from pprint import pprint

import torch
import torchmetrics.classification

from charmory.tasks.base import BaseEvaluationTask


class ImageClassificationTask(BaseEvaluationTask):
    """Image classification evaluation task"""

    def __init__(self, *args, num_classes: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )

    def run_benign(self, batch: BaseEvaluationTask.Batch):
        super().run_benign(batch)
        self.accuracy(torch.tensor(batch.y_pred), torch.tensor(batch.y))

    def on_test_end(self):
        pprint(self.accuracy.compute())
