from typing import Optional

from charmory.data import (
    Accessor,
    Batch,
    BatchedImages,
    DataType,
    ImageDimensions,
    Scale,
)
from charmory.export.base import Exporter


class ImageClassificationExporter(Exporter):
    def __init__(
        self,
        inputs_accessor: Optional[BatchedImages.Accessor] = None,
        predictions_accessor: Optional[Accessor] = None,
        targets_accessor: Optional[Accessor] = None,
    ):
        super().__init__(
            predictions_accessor=predictions_accessor, targets_accessor=targets_accessor
        )
        self.inputs_accessor = inputs_accessor or BatchedImages.as_numpy(
            dim=ImageDimensions.HWC, scale=Scale(dtype=DataType.FLOAT, max=1.0)
        )

    def export(self, chain_name: str, batch_idx: int, batch: Batch) -> None:
        assert self.sink, "No sink has been set, unable to export"

        self._export_metadata(chain_name, batch_idx, batch)

        images = self.inputs_accessor.get(batch.inputs)
        for sample_idx, image in enumerate(images):
            filename = f"batch_{batch_idx}_ex_{sample_idx}_{chain_name}.png"
            self.sink.log_image(image, filename)
