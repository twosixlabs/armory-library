from typing import Iterable, Optional

from armory.data import (
    DataSpecification,
    DataType,
    ImageClassificationBatch,
    ImageDimensions,
    NumpyImageSpec,
    Scale,
)
from armory.export.base import Exporter


class ImageClassificationExporter(Exporter):
    """An exporter for image classification samples."""

    def __init__(
        self,
        name: Optional[str] = None,
        inputs_spec: Optional[NumpyImageSpec] = None,
        predictions_spec: Optional[DataSpecification] = None,
        targets_spec: Optional[DataSpecification] = None,
        criterion: Optional[Exporter.Criterion] = None,
    ):
        """
        Initializes the exporter.

        Args:
            name: Descriptive name of the exporter
            inputs_spec: Optional, data specification used to obtain raw
                image data from the inputs contained in exported batches. By
                default, a NumPy images specification is used.
            predictions_spec: Optional, data specification used to obtain raw
                predictions data from the exported batches. By default, a generic
                NumPy specification is used.
            targets_spec: Optional, data specification used to obtain raw ground
                truth targets data from the exported batches. By default, a
                generic NumPy specification is used.
            criterion: Criterion to determine when samples will be exported. If
                omitted, no samples will be exported.
        """
        super().__init__(
            name=name or "ImageClassification",
            predictions_spec=predictions_spec,
            targets_spec=targets_spec,
            criterion=criterion,
        )
        self.inputs_spec = inputs_spec or NumpyImageSpec(
            dim=ImageDimensions.HWC, scale=Scale(dtype=DataType.FLOAT, max=1.0)
        )

    def export_samples(
        self, batch_idx: int, batch: ImageClassificationBatch, samples: Iterable[int]
    ) -> None:
        assert self.sink, "No sink has been set, unable to export"
        self._export_metadata(batch_idx, batch, samples)
        images = batch.inputs.get(self.inputs_spec)
        for sample_idx in samples:
            filename = self.artifact_path(batch_idx, sample_idx, "input.png")
            self.sink.log_image(images[sample_idx], filename)
