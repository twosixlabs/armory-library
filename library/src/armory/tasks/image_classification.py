"""Image classification evaluation task"""

from armory.tasks.base import BaseEvaluationTask


class ImageClassificationTask(BaseEvaluationTask):
    """Image classification evaluation task"""

    def export_batch(self, batch: BaseEvaluationTask.Batch):
        if batch.x_perturbed is not None:
            self._export_image(batch.chain_name, batch.x_perturbed, batch.i)
        self.export_batch_metadata(batch)

    def _export_image(self, chain_name, batch_data, batch_idx):
        batch_size = batch_data.shape[0]
        for sample_idx in range(batch_size):
            filename = f"batch_{batch_idx}_ex_{sample_idx}_{chain_name}.png"
            image = batch_data[sample_idx]
            if self.export_adapter is not None:
                image = self.export_adapter(image)
            self.exporter.log_image(image, filename)
