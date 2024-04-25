from typing import Iterable, Optional

from armory.data import Accessor, Batch, DataType, ImageDimensions, Images, Scale
from armory.export.base import Exporter


class ImageClassificationExporter(Exporter):
    """An exporter for image classification samples."""

    def __init__(
        self,
        inputs_accessor: Optional[Images.Accessor] = None,
        predictions_accessor: Optional[Accessor] = None,
        targets_accessor: Optional[Accessor] = None,
        criterion: Optional[Exporter.Criterion] = None,
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
            criterion: Criterion dictating when samples will be exported. If
                omitted, no samples will be exported.
        """
        super().__init__(
            predictions_accessor=predictions_accessor,
            targets_accessor=targets_accessor,
            criterion=criterion,
        )
        self.inputs_accessor = inputs_accessor or Images.as_numpy(
            dim=ImageDimensions.HWC, scale=Scale(dtype=DataType.FLOAT, max=1.0)
        )

    def export_samples(
        self, batch_idx: int, batch: Batch, samples: Iterable[int]
    ) -> None:
        assert self.sink, "No sink has been set, unable to export"
        self._export_metadata(batch_idx, batch, samples)
        images = self.inputs_accessor.get(batch.inputs)
        for sample_idx in samples:
            filename = self.artifact_path(batch_idx, sample_idx, "input.png")
            self.sink.log_image(images[sample_idx], filename)
