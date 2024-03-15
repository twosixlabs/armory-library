from typing import Iterable, Optional, Sequence, Set, Union

import torch

from armory.export.base import Exporter


def always() -> Exporter.Criteria:
    def _criteria(chain_name, batch_idx, batch):
        return True

    return _criteria


def _to_set(value: Union[bool, Iterable[int]], batch) -> Set[int]:
    if not value:
        return set()
    if type(value) is bool:
        return set(range(len(batch)))
    return set(value)


def all_satisfied(*criteria: Exporter.Criteria) -> Exporter.Criteria:
    def _criteria(chain_name, batch_idx, batch):
        aggregate_to_export: Optional[Set[int]] = None
        for c in criteria:
            to_export = _to_set(c(chain_name, batch_idx, batch), batch)
            if aggregate_to_export is None:
                aggregate_to_export = to_export
            else:
                aggregate_to_export.intersection_update(to_export)
        return aggregate_to_export if aggregate_to_export is not None else set()

    return _criteria


def any_satisfied(*criteria: Exporter.Criteria) -> Exporter.Criteria:
    def _criteria(chain_name, batch_idx, batch):
        aggregate_to_export: Optional[Set[int]] = None
        for c in criteria:
            to_export = _to_set(c(chain_name, batch_idx, batch), batch)
            if aggregate_to_export is None:
                aggregate_to_export = to_export
            else:
                aggregate_to_export.update(to_export)
        return aggregate_to_export if aggregate_to_export is not None else set()

    return _criteria


def not_criteria(criteria: Exporter.Criteria) -> Exporter.Criteria:
    def _criteria(chain_name, batch_idx, batch):
        res = criteria(chain_name, batch_idx, batch)
        if not res:
            return True
        if type(res) is bool:
            return False
        return set(range(len(batch))).difference(res)

    return _criteria


def every_n_batches(n: int) -> Exporter.Criteria:
    def _criteria(chain_name, batch_idx: int, batch):
        if n == 0:
            return False
        return (batch_idx + 1) % n == 0

    return _criteria


def first_n_batches(n: int) -> Exporter.Criteria:
    def _criteria(chain_name, batch_idx: int, batch):
        if n == 0:
            return False
        return batch_idx < n

    return _criteria


def every_n_samples_of_batch(n: int) -> Exporter.Criteria:
    def _criteria(chain_name, batch_idx: int, batch):
        if n == 0:
            return False
        return [
            sample_idx for sample_idx in range(len(batch)) if (sample_idx + 1) % n == 0
        ]

    return _criteria


def every_n_samples(n: int) -> Exporter.Criteria:
    def _criteria(chain_name, batch_idx: int, batch):
        if n == 0:
            return False
        batch_size = len(batch)
        return [
            sample_idx
            for sample_idx in range(batch_size)
            if ((batch_idx * batch_size) + sample_idx + 1) % n == 0
        ]

    return _criteria


def first_n_samples_of_batch(n: int) -> Exporter.Criteria:
    def _criteria(chain_name, batch_idx: int, batch):
        if n == 0:
            return False
        return [sample_idx for sample_idx in range(len(batch)) if sample_idx < n]

    return _criteria


def first_n_samples(n: int) -> Exporter.Criteria:
    def _criteria(chain_name, batch_idx: int, batch):
        if n == 0:
            return False
        batch_size = len(batch)
        if batch_idx * batch_size >= n:
            return False
        return [
            sample_idx
            for sample_idx in range(batch_size)
            if ((batch_idx * batch_size) + sample_idx) < n
        ]

    return _criteria


def chains(names: Sequence[str]) -> Exporter.Criteria:
    def _criteria(chain_name: str, batch_idx, batch):
        return chain_name in names

    return _criteria


def samples(indices: Sequence[int]) -> Exporter.Criteria:
    def _criteria(chain_name, batch_idx: int, batch):
        if len(indices) == 0:
            return False
        batch_size = len(batch)
        return [
            sample_idx
            for sample_idx in range(batch_size)
            if ((batch_idx * batch_size) + sample_idx) in indices
        ]

    return _criteria


def when_metric_eq(metric, threshold) -> Exporter.Criteria:
    def _criteria(chain_name, batch_idx, batch):
        val = metric(batch)
        res = val == threshold
        if type(res) == torch.Tensor:
            if res.dim() == 0:
                return res.item()
            return res.nonzero().flatten().tolist()
        return res

    return _criteria


def when_metric_lt(metric, threshold) -> Exporter.Criteria:
    def _criteria(chain_name, batch_idx, batch):
        val = metric(batch)
        res = val < threshold
        if type(res) == torch.Tensor:
            if res.dim() == 0:
                return res.item()
            return res.nonzero().flatten().tolist()
        return res

    return _criteria


def when_metric_gt(metric, threshold) -> Exporter.Criteria:
    def _criteria(chain_name, batch_idx, batch):
        val = metric(batch)
        res = val > threshold
        if type(res) == torch.Tensor:
            if res.dim() == 0:
                return res.item()
            return res.nonzero().flatten().tolist()
        return res

    return _criteria
