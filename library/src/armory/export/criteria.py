"""Export criterion utilities"""

import math
from typing import Callable, Iterable, Optional, Sequence, Set, Union

import torch

from armory.data import Batch, DataSpecification, TorchSpec
from armory.export.base import Exporter


def always() -> Exporter.Criterion:
    """
    Creates an export criterion that matches all samples of all batches.

    Example::

        from armory.export import Exporter
        from armory.export.criteria import always

        exporter = Exporter(criterion=always())

    Returns:
        Export criterion function
    """

    def _criterion(batch_idx, batch):
        return True

    return _criterion


def _to_set(value: Union[bool, Iterable[int]], batch) -> Set[int]:
    if not value:
        return set()
    if type(value) is bool:
        return set(range(len(batch)))
    return set(value)


def all_satisfied(*criteria: Exporter.Criterion) -> Exporter.Criterion:
    """
    Creates an export criterion that matches samples that satisfy all of the given
    nested criteria.

    Example::

        from armory.export import Exporter
        from armory.export.criteria import all_satisfied, every_n_samples, first_n_batches

        # Only exports every other sample from the first 2 batches
        exporter = Exporter(
            criterion=all_satisfied(
                every_n_samples(2),
                first_n_batches(2),
            )
        )

    Args:
        *criteria: Nested criteria that must all be satisfied for a sample to
            be exported

    Returns:
        Export criterion function
    """

    def _criterion(batch_idx, batch):
        aggregate_to_export: Optional[Set[int]] = None
        for c in criteria:
            to_export = _to_set(c(batch_idx, batch), batch)
            if aggregate_to_export is None:
                aggregate_to_export = to_export
            else:
                aggregate_to_export.intersection_update(to_export)
                # If all samples have been disqualified, no need to continue
                # evaluating the remaining criteria
                if not aggregate_to_export:
                    return aggregate_to_export
        return aggregate_to_export if aggregate_to_export is not None else set()

    return _criterion


def any_satisfied(*criteria: Exporter.Criterion) -> Exporter.Criterion:
    """
    Creates an export criterion that matches samples that satisfy any of the given
    nested criteria.

    Example::

        from armory.export import Exporter
        from armory.export.criteria import any_satisfied, every_n_samples, first_n_batches

        # Exports every sample from the first 2 batches, then every other sample
        # for all other batches
        exporter = Exporter(
            criterion=any_satisfied(
                every_n_samples(2),
                first_n_batches(2),
            )
        )

    Args:
        *criteria: Nested criteria of which at least one must be satisfied for
            a sample to be exported

    Returns:
        Export criterion function
    """

    def _criterion(batch_idx, batch):
        aggregate_to_export: Set[int] = set()
        for c in criteria:
            to_export = _to_set(c(batch_idx, batch), batch)
            aggregate_to_export.update(to_export)
        return aggregate_to_export

    return _criterion


def not_satisfied(criterion: Exporter.Criterion) -> Exporter.Criterion:
    """
    Creates an export criterion that matches samples that do not satisfy the
    nested criterion.

    Example::

        from armory.export import Exporter
        from armory.export.criteria import every_n_batches, not_satisfied

        # Exports every sample in every batch except every 3rd batch (when
        # batch_idx is 0, 1, 3, etc.)
        exporter = Exporter(criterion=not_satisfied(every_n_batches(3)))

    Args:
        criterion: Nested criterion for which unsatisfied samples are to be
            exported

    Returns:
        Export criterion function
    """

    def _criterion(batch_idx, batch):
        res = criterion(batch_idx, batch)
        if not res:
            return True
        if type(res) is bool:
            return False
        return set(range(len(batch))).difference(res)

    return _criterion


def every_n_batches(n: int) -> Exporter.Criterion:
    """
    Creates an export criterion that matches all samples from every nth batch.

    Note, if the data loader is shuffling the dataset, the samples that get
    exported will vary between runs.

    Example::

        from armory.export import Exporter
        from armory.export.criteria import every_n_batches

        # Exports every sample in every 5th batch (when batch_idx is 4, 9, 14, etc.)
        exporter = Exporter(criterion=every_n_batches(5))

    Args:
        n: Interval at which to export batches based on the index of the batch

    Returns:
        Export criterion function
    """

    def _criterion(batch_idx: int, batch):
        if n == 0:
            return False
        return (batch_idx + 1) % n == 0

    return _criterion


def first_n_batches(n: int) -> Exporter.Criterion:
    """
    Creates an export criterion that matches all samples from the first n batches.

    Note, if the data loader is shuffling the dataset, the samples that get
    exported will vary between runs.

    Example::

        from armory.export import Exporter
        from armory.export.criteria import first_n_batches

        # Exports every sample in the first 3 batches (when batch_idx is 0, 1, and 2)
        exporter = Exporter(criterion=every_n_batches(3))

    Args:
        n: Number of batches to be exported

    Returns:
        Export criterion function
    """

    def _criterion(batch_idx: int, batch):
        if n == 0:
            return False
        return batch_idx < n

    return _criterion


def every_n_samples_of_batch(n: int) -> Exporter.Criterion:
    """
    Creates an export criterion that matches every nth sample in every batch.

    Note, if the data loader is shuffling the dataset, the samples that get
    exported will vary between runs.

    Example::

        from armory.export import Exporter
        from armory.export.criteria import every_n_samples_of_batch

        # Exports every other sample in every batch (when sample_idx is 1, 3, 5, etc.)
        exporter = Exporter(criterion=every_n_samples_of_batch(5))

    Args:
        n: Interval at which to export samples based on the index of the sample
            within each batch

    Returns:
        Export criterion function
    """

    def _criterion(batch_idx: int, batch):
        if n == 0:
            return False
        return (
            sample_idx for sample_idx in range(len(batch)) if (sample_idx + 1) % n == 0
        )

    return _criterion


def every_n_samples(n: int) -> Exporter.Criterion:
    """
    Creates an export criterion that matches every nth sample in the dataset.

    Note, if the data loader is shuffling the dataset, the samples that get
    exported will vary between runs.

    Example::

        from armory.export import Exporter
        from armory.export.criteria import every_n_samples

        # Exports every 3rd sample, regardless of batch
        # For example with a batch size of 4:
        #  - batch 0, sample 2
        #  - batch 1, sample 1
        #  - batch 2, samples 0 and 3
        exporter = Exporter(criterion=every_n_samples(3))

    Args:
        n: Interval at which to export samples based on the index of the sample
            within the dataset

    Returns:
        Export criterion function
    """

    def _criterion(batch_idx: int, batch):
        if n == 0:
            return False
        batch_size = len(batch)
        return (
            sample_idx
            for sample_idx in range(batch_size)
            if ((batch_idx * batch_size) + sample_idx + 1) % n == 0
        )

    return _criterion


def first_n_samples_of_batch(n: int) -> Exporter.Criterion:
    """
    Creates an export criterion that matches the first n samples in every batch.

    Note, if the data loader is shuffling the dataset, the samples that get
    exported will vary between runs.

    Example::

        from armory.export import Exporter
        from armory.export.criteria import first_n_samples_of_batch

        # Exports the first 2 samples in every batches (when sample_idx is 0 and 1)
        exporter = Exporter(criterion=first_n_samples_of_batch(2))

    Args:
        n: Number of samples to be exported from each batch

    Returns:
        Export criterion function
    """

    def _criterion(batch_idx: int, batch):
        if n == 0:
            return False
        return (sample_idx for sample_idx in range(len(batch)) if sample_idx < n)

    return _criterion


def first_n_samples(n: int) -> Exporter.Criterion:
    """
    Creates an export criterion that matches the first n samples in the dataset.

    Note, if the data loader is shuffling the dataset, the samples that get
    exported will vary between runs.

    Example::

        from armory.export import Exporter
        from armory.export.criteria import first_n_samples

        # Exports the first 5 samples, regardless of batch
        # For example with a batch size of 2:
        #  - batch 0, samples 0 and 1
        #  - batch 1, samples 0 and 1
        #  - batch 2, sample 0
        exporter = Exporter(criterion=first_n_samples(5))

    Args:
        n: Number of samples to be exported

    Returns:
        Export criterion function
    """

    def _criterion(batch_idx: int, batch):
        if n == 0:
            return False
        batch_size = len(batch)
        if batch_idx * batch_size >= n:
            return False
        return (
            sample_idx
            for sample_idx in range(batch_size)
            if ((batch_idx * batch_size) + sample_idx) < n
        )

    return _criterion


def samples(indices: Sequence[int]) -> Exporter.Criterion:
    """
    Creates an export criterion that matches specific samples in the dataset, by
    their global index (regardless of batch).

    Note, if the data loader is shuffling the dataset, the samples that get
    exported will vary between runs.

    Example::

        from armory.export import Exporter
        from armory.export.criteria import samples

        # For a batch size of 4:
        #  - batch 0, sample 2
        #  - batch 1, samples 0 and 1
        #  - batch 2, samples 2 and 3
        exporter = Exporter(criterion=samples([2, 4, 5, 10, 11]))

    Args:
        indices: Indices of samples within the dataset to be exported

    Returns:
        Export criterion function
    """

    def _criterion(batch_idx: int, batch):
        if len(indices) == 0:
            return False
        batch_size = len(batch)
        return (
            sample_idx
            for sample_idx in range(batch_size)
            if ((batch_idx * batch_size) + sample_idx) in indices
        )

    return _criterion


def _create_metric_criterion(comp, metric, threshold) -> Exporter.Criterion:
    def _criterion(batch_idx, batch):
        val = metric(batch)
        res = comp(val, threshold)
        if type(res) == torch.Tensor:
            if res.dim() == 0:
                return res.item()
            return res.nonzero().flatten().tolist()
        return res

    return _criterion


def when_metric_eq(
    metric: Callable[[Batch], Union[bool, float, torch.Tensor]],
    threshold: Union[bool, float, torch.Tensor],
) -> Exporter.Criterion:
    """
    Creates an export criterion that matches when a computed metric for a batch
    or the samples within the batch is equal to a particular value.

    Example::

        import torch
        from armory.data import TorchSpec
        from armory.export import Exporter
        from armory.export.criteria import when_metric_eq

        # Exports samples that have max score of exactly 5
        def max_pred(batch):
            return torch.tensor([
                torch.max(p) for p in batch.predictions.get(TorchSpec())
            ])
        exporter = Exporter(criterion=when_metric_eq(max_pred, 5))

    Args:
        metric: Callable that computes a metric for a batch. The return value
            may be a single boolean or number, or it can be a tensor array of
            the computed metric values for each sample in the batch.
        threshold: Value the computed metric (either batchwise or samplewise)
            must be equal to in order for the batch or samples to be exported

    Returns:
        Export criterion function
    """
    return _create_metric_criterion(lambda lhs, rhs: lhs == rhs, metric, threshold)


def when_metric_isclose(
    metric: Callable[[Batch], Union[float, torch.Tensor]],
    threshold: Union[float, torch.Tensor],
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> Exporter.Criterion:
    """
    Creates an export criterion that matches when a computed metric for a batch
    or the samples within the batch is close to a particular value.

    Example::

        import torch
        from armory.data import TorchSpec
        from armory.export import Exporter
        from armory.export.criteria import when_metric_isclose

        # Exports samples that have max score of 5.0
        def max_pred(batch):
            return torch.tensor([
                torch.max(p) for p in batch.predictions.get(TorchSpec())
            ])
        exporter = Exporter(criterion=when_metric_isclose(max_pred, 5))

    Args:
        metric: Callable that computes a metric for a batch. The return value
            may be a single boolean or number, or it can be a tensor array of
            the computed metric values for each sample in the batch.
        threshold: Value the computed metric (either batchwise or samplewise)
            must be equal to in order for the batch or samples to be exported
        rtol: Optional, relative tolerance
        atol: Optional, absolute tolerance

    Returns:
        Export criterion function
    """

    def isclose(lhs, rhs):
        if isinstance(lhs, torch.Tensor):
            if not isinstance(rhs, torch.Tensor):
                rhs = torch.as_tensor(rhs, dtype=lhs.dtype)
            return torch.isclose(lhs, rhs, rtol=rtol, atol=atol)
        else:
            return math.isclose(lhs, rhs, rel_tol=rtol, abs_tol=atol)

    return _create_metric_criterion(isclose, metric, threshold)


def when_metric_lt(
    metric: Callable[[Batch], Union[float, torch.Tensor]],
    threshold: Union[float, torch.Tensor],
) -> Exporter.Criterion:
    """
    Creates an export criterion that matches when a computed metric for a batch
    or the samples within the batch is less than a particular threshold value.

    Example::

        import torch
        from armory.data import TorchSpec
        from armory.export import Exporter
        from armory.export.criteria import when_metric_lt

        # Exports samples that have max score less than 5
        def max_pred(batch):
            return torch.tensor([
                torch.max(p) for p in batch.predictions.get(TorchSpec())
            ])
        exporter = Exporter(criterion=when_metric_lt(max_pred, 5))

    Args:
        metric: Callable that computes a metric for a batch. The return value
            may be a single number, or it can be a tensor array of the computed
            metric values for each sample in the batch.
        threshold: Value the computed metric (either batchwise or samplewise)
            must be less than in order for the batch or samples to be exported

    Returns:
        Export criterion function
    """
    return _create_metric_criterion(lambda lhs, rhs: lhs < rhs, metric, threshold)


def when_metric_gt(metric, threshold) -> Exporter.Criterion:
    """
    Creates an export criterion that matches when a computed metric for a batch
    or the samples within the batch is greater than a particular threshold value.

    Example::

        import torch
        from armory.data import TorchSpec
        from armory.export import Exporter
        from armory.export.criteria import when_metric_gt

        # Exports samples that have max score greater than 5
        def max_pred(batch):
            return torch.tensor([
                torch.max(p) for p in batch.predictions.get(TorchSpec())
            ])
        exporter = Exporter(criterion=when_metric_gt(max_pred, 5))

    Args:
        metric: Callable that computes a metric for a batch. The return value
            may be a single number, or it can be a tensor array of the computed
            metric values for each sample in the batch.
        threshold: Value the computed metric (either batchwise or samplewise)
            must be greater than in order for the batch or samples to be exported

    Returns:
        Export criterion function
    """
    return _create_metric_criterion(lambda lhs, rhs: lhs > rhs, metric, threshold)


def when_metric_in(
    metric: Callable[[Batch], Union[float, torch.Tensor]],
    threshold: Union[Sequence[float], Sequence[torch.Tensor]],
) -> Exporter.Criterion:
    """
    Creates an export criterion that matches when a computed metric for a batch
    or the samples within the batch is one of a particular set of values.

    Example::

        import torch
        from armory.data import TorchSpec
        from armory.export import Exporter
        from armory.export.criteria import when_metric_in

        # Exports samples that have max score of 5 or 8
        def max_pred(batch):
            return torch.tensor([
                torch.max(p) for p in batch.predictions.get(TorchSpec())
            ])
        exporter = Exporter(criterion=when_metric_in(max_pred, [5, 8]))

    Args:
        metric: Callable that computes a metric for a batch. The return value
            may be a single boolean or number, or it can be a tensor array of
            the computed metric values for each sample in the batch.
        threshold: Possible values the computed metric (either batchwise or
            samplewise) must be equal to in order for the batch or samples to
            be exported

    Returns:
        Export criterion function
    """

    def _comp(lhs, rhs):
        if isinstance(lhs, torch.Tensor):
            return lhs.clone().apply_(lambda x: x in rhs).bool()
        return lhs in rhs

    return _create_metric_criterion(_comp, metric, threshold)


def batch_targets(
    spec: Optional[DataSpecification] = None,
) -> Callable[[Batch], torch.Tensor]:
    """
    Creates a batch metric callable that returns the targets from the batch.

    Example::

        from armory.export import Exporter
        from armory.export.criteria import batch_targets, when_metric_lt

        # Exports samples that have a target value less than 10
        exporter = Exporter(criterion=when_metric_lt(batch_targets(), 10))

    Args:
        spec: Data specification for obtaining targets in a batch

    Returns:
        Batch metric function
    """
    if spec is None:
        spec = TorchSpec()

    def _metric(batch):
        return batch.targets.get(spec)

    return _metric
