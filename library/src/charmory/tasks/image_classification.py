"""Image classification evaluation task"""

from charmory.tasks.base import BaseEvaluationTask


class ImageClassificationTask(BaseEvaluationTask):
    """Image classification evaluation task"""

    def export_batch(
        self, chain_name: str, batch_idx: int, batch  #: BaseEvaluationTask.Batch
    ):
        # if batch.was_perturbed
        # if batch.x_perturbed is not None:
        self._export_image(chain_name, batch_idx, batch.inputs.numpy(dim=HWC))
        self.export_batch_metadata(batch)

    def _export_image(self, chain_name, batch_idx, images):
        for sample_idx, image in enumerate(images):
            filename = f"batch_{batch_idx}_ex_{sample_idx}_{chain_name}.png"
            image = batch_data[sample_idx]
            if self.export_adapter is not None:
                image = self.export_adapter(image)
            self.exporter.log_image(image, filename)
