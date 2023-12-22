from typing import TYPE_CHECKING, Optional

from charmory.batch import BatchedImages, ImageDimensions
from charmory.export.base import Exporter

if TYPE_CHECKING:
    from charmory.batch import Batch


class ImageClassificationExporter(Exporter):
    def __init__(self, accessor: Optional[BatchedImages.Accessor] = None):
        super().__init__()
        self.accessor = accessor or BatchedImages.as_numpy(dim=ImageDimensions.HWC)

    def export(self, chain_name: str, batch_idx: int, batch: "Batch") -> None:
        self._export_metadata(chain_name, batch_idx, batch)

        images = self.accessor.get(batch.inputs)
        for sample_idx, image in enumerate(images):
            filename = f"batch_{batch_idx}_ex_{sample_idx}_{chain_name}.png"
            self.sink.log_image(image, filename)
