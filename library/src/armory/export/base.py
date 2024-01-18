from abc import ABC, abstractmethod
from typing import Mapping, Optional

from armory.data import Accessor, Batch, DefaultNumpyAccessor
from armory.export.sink import Sink


class Exporter(ABC):
    def __init__(
        self,
        predictions_accessor: Optional[Accessor] = None,
        targets_accessor: Optional[Accessor] = None,
    ) -> None:
        self.predictions_accessor = predictions_accessor or DefaultNumpyAccessor()
        self.targets_accessor = targets_accessor or DefaultNumpyAccessor()
        self.sink: Optional[Sink] = None

    def use_sink(self, sink: Sink) -> None:
        self.sink = sink

    @abstractmethod
    def export(self, chain_name: str, batch_idx: int, batch: Batch) -> None:
        ...

    @staticmethod
    def _from_list(maybe_list, idx):
        try:
            return maybe_list[idx]
        except:  # noqa: E722
            # if it's None or is not a list/sequence/etc, just return None
            return None

    def _export_metadata(self, chain_name: str, batch_idx: int, batch: Batch) -> None:
        assert self.sink, "No sink has been set, unable to export"

        targets = self.targets_accessor.get(batch.targets)
        predictions = self.predictions_accessor.get(batch.predictions)

        for sample_idx in range(len(batch)):
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
                artifact_file=f"batch_{batch_idx}_ex_{sample_idx}_{chain_name}.txt",
            )


class NullExporter(Exporter):
    def export(self, chain_name: str, batch_idx: int, batch: Batch) -> None:
        pass
