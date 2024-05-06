from abc import ABC, abstractmethod
from typing import Callable, Iterable, Mapping, Optional, Union

from armory.data import Batch, DataSpecification, NumpySpec
from armory.export.sink import Sink
from armory.track import Trackable


class Exporter(Trackable, ABC):
    """Base class for an Armory sample exporter."""

    Criterion = Callable[[int, Batch], Union[bool, Iterable[int]]]

    def __init__(
        self,
        name: str,
        predictions_spec: Optional[DataSpecification] = None,
        targets_spec: Optional[DataSpecification] = None,
        criterion: Optional[Criterion] = None,
    ) -> None:
        """
        Initializes the exporter.

        Args:
            name: Descriptive name of the exporter
            predictions_spec: Optional, data specification used to obtain raw
                predictions data from the exported batches. By default, a generic
                NumPy specification will be used.
            targets_spec: Optional, data specification used to obtain raw ground
                truth targets data from the exported batches. By default, a
                generic NumPy specification is used.
            criterion: Criterion to determine when samples will be exported. If
                omitted, no samples will be exported.
        """
        super().__init__()
        self.name = name
        self.predictions_spec = predictions_spec or NumpySpec()
        self.targets_spec = targets_spec or NumpySpec()
        self.sink: Optional[Sink] = None
        self.criterion = criterion

    def use_sink(self, sink: Sink) -> None:
        """Sets the export sink to be used by the exporter."""
        self.sink = sink

    def export(self, batch_idx: int, batch: Batch) -> None:
        """
        Exports the given batch.

        Args:
            batch_idx: The index/number of this batch.
            batch: The batch to be exported.
        """
        assert self.sink, "No sink has been set, unable to export"
        if self.criterion is None:
            return
        to_export = self.criterion(batch_idx, batch)
        if not to_export:
            return
        if type(to_export) is bool:
            # Because of the early-return above, to_export can only ever be True at this point
            to_export = range(len(batch))
        self.export_samples(batch_idx, batch, to_export)

    @abstractmethod
    def export_samples(
        self, batch_idx: int, batch: Batch, samples: Iterable[int]
    ) -> None:
        """
        Exports samples from the given batch.

        Args:
            batch_idx: The index/number of this batch.
            batch: The batch to be exported.
            samples: The indices of samples in the batch to be exported.
        """
        ...

    @staticmethod
    def artifact_path(batch_idx: int, sample_idx: int, filename: str) -> str:
        """
        Creates the full artifact path for a particular sample export.

        Args:
            batch_idx: The index/number of the sample's batch.
            sample_idx: The index/number of the sample within the batch.
            filename: The name of the exported file.

        Returns:
            Full artifact path as a string.
        """
        return f"exports/{batch_idx:05}/{sample_idx:02}/{filename}"

    @staticmethod
    def _from_list(maybe_list, idx):
        try:
            return maybe_list[idx]
        except:  # noqa: E722
            # if it's None or is not a list/sequence/etc, just return None
            return None

    def _export_metadata(
        self, batch_idx: int, batch: Batch, samples: Iterable[int]
    ) -> None:
        assert self.sink, "No sink has been set, unable to export"

        targets = batch.targets.get(self.targets_spec)
        predictions = batch.predictions.get(self.predictions_spec)

        for sample_idx in samples:
            dictionary = dict(
                targets=self._from_list(targets, sample_idx),
                predictions=self._from_list(predictions, sample_idx),
            )
            for key, value in batch.metadata["data"].items():
                dictionary[key] = self._from_list(value, sample_idx)
            for perturbation, metadata in batch.metadata["perturbations"].items():
                if isinstance(metadata, Mapping):
                    dictionary.update(
                        {
                            f"{perturbation}.{k}": self._from_list(v, sample_idx)
                            for k, v in metadata.items()
                        }
                    )
                else:
                    dictionary[perturbation] = self._from_list(metadata, sample_idx)

            self.sink.log_dict(
                dictionary=dictionary,
                artifact_file=self.artifact_path(batch_idx, sample_idx, "metadata.txt"),
            )
