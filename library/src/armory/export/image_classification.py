from typing import Optional

from armory.data import Accessor, Batch, DataType, ImageDimensions, Images, Scale
from armory.export.base import Exporter


class ImageClassificationExporter(Exporter):
    """An exporter for image classification samples."""

    def __init__(
        self,
        inputs_accessor: Optional[Images.Accessor] = None,
        predictions_accessor: Optional[Accessor] = None,
        targets_accessor: Optional[Accessor] = None,
    ):
        """
        Initializes the exporter.

        Args:
            inputs_accessor: Optional, data exporter used to obtain low-level
                image data from the highly-structured inputs contained in
                exported batches. By default, a NumPy images accessor is used.
            predictions_accessor: Optional, data exporter used to obtain
                low-level predictions data from the highly-structured
                predictions contained in exported batches. By default, a generic
                NumPy accessor is used.
            targets_accessor: Optional, data exporter used to obtain low-level
                ground truth targets data from the high-ly structured targets
                contained in exported batches. By default, a generic NumPy
                accessor is used.
        """
        super().__init__(
            predictions_accessor=predictions_accessor, targets_accessor=targets_accessor
        )
        self.inputs_accessor = inputs_accessor or Images.as_numpy(
            dim=ImageDimensions.HWC, scale=Scale(dtype=DataType.FLOAT, max=1.0)
        )

    def export(self, chain_name: str, batch_idx: int, batch: Batch) -> None:
        assert self.sink, "No sink has been set, unable to export"

        self._export_metadata(chain_name, batch_idx, batch)

        images = self.inputs_accessor.get(batch.inputs)
        for sample_idx, image in enumerate(images):
            filename = f"batch_{batch_idx}_ex_{sample_idx}_{chain_name}.png"
            self.sink.log_image(image, filename)
