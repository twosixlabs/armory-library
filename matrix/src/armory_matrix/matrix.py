import concurrent.futures
from copy import deepcopy
import itertools
from typing import Any, Callable, Iterable, Mapping, Optional


def create_matrix(worker_num: Optional[int] = None, num_workers: Optional[int] = None):
    def _create(**kwargs) -> Iterable[Mapping[str, Any]]:
        # Keep ordered lists of keys and value iterables
        keys = []
        values = []
        for key, value in kwargs.items():
            keys.append(key)
            if isinstance(value, Iterable):
                values.append(value)
            else:
                values.append([value])

        # Create cartesian product of all possible parameter values
        product = itertools.product(*values)

        # Get subset of rows if running parallelized
        if worker_num is not None and num_workers is not None:
            product = list(product)
            num_rows = len(product)
            # // is the floor (or integer) division operator
            start = worker_num * num_rows // num_workers
            stop = (worker_num + 1) * num_rows // num_workers
            product = product[start:stop]

        matrix = []
        for values in product:
            # Create a key-value mapping for each argument in each row
            matrix.append({k: v for k, v in zip(keys, values)})

        return matrix

    return _create


class Matrix:
    def __init__(self, func, **kwargs):
        self.func = func
        self.kwargs = kwargs
        self.worker_num: Optional[int] = None
        self.num_workers: Optional[int] = None
        self.pruner: Optional[Callable[..., bool]] = None

    @property
    def matrix(self):
        return create_matrix(
            worker_num=self.worker_num,
            num_workers=self.num_workers,
        )(**self.kwargs)

    def __call__(self, *args, **kwargs):
        results = []
        for it in self.matrix:
            it_args = deepcopy(args)
            it_kwargs = deepcopy(kwargs)
            it_kwargs.update(it)
            if self.pruner and self.pruner(*it_args, **it_kwargs):
                continue
            results.append(self.func(*it_args, **it_kwargs))
        return results

    def partition(self, worker_num: Optional[int], num_workers: Optional[int]):
        self.worker_num = worker_num
        self.num_workers = num_workers
        return self

    def prune(self, pruner: Optional[Callable[..., bool]]):
        self.pruner = pruner
        return self

    def parallel(self, max_workers):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        def _run(*args, **kwargs):
            futures = []
            for it in self.matrix:
                it_args = deepcopy(args)
                it_kwargs = deepcopy(kwargs)
                it_kwargs.update(it)
                if self.pruner and self.pruner(*it_args, **it_kwargs):
                    continue
                futures.append(executor.submit(self.func, *it_args, **it_kwargs))

            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
            return results

        return _run


def matrix(**kwargs):
    def decorator(func):
        return Matrix(func, **kwargs)

    return decorator
