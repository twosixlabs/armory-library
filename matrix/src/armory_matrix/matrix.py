"""Matrix generation utilities"""

import concurrent.futures
from copy import deepcopy
import itertools
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple


def create_matrix(
    worker_num: Optional[int] = None,
    num_workers: Optional[int] = None,
    prune: Optional[Callable[..., bool]] = None,
):
    """
    Creates a matrix of variable parameter values as a sequence of key-value
    argument mappings. The matrix is created as the cartesian product of all
    possible argument combinations.

    This function is not intended to be used directly, but is used internally
    by the `Matrix` class.

    Example::

        >>> from armory_matrix.matrix import create_matrix
        >>> create_matrix()(a=[1, 2], b=[3, 4])
        [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}, {'a': 2, 'b': 4}]
        >>> create_matrix(1, 2)(a=[1, 2], b=[3, 4])
        [{'a': 2, 'b': 3}, {'a': 2, 'b': 4}]
        >>> create_matrix(None, None, lambda a, b: a=="2" and b=="4")(a=[1, 2], b=[3, 4])
        [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}]

    Args:
        worker_num: The index of this node among all nodes, used to determine
            which partition of the matrix to return. If specified, `num_workers`
            must also be provided.
        num_workers: The count of all nodes, used to determine the size of the
            partition of the matrix to return. If specified, `worker_num` must
            also be provided.
        prune: A callable to determine whether a row of the matrix should be
            pruned. If the callable returns `True` for a given set of
            parameters, that parameter set, or row, will be omitted from the
            returned matrix.

    Returns:
        A function that will generate a sequence of key-value mappings. The
        function accepts keyword arguments of iterables, ranges, or sequences
        specifying the allowable values for each argument. The return value of
        the function is a sequence of key-value mappings that are the arguments
        corresponding to each row of the matrix.
    """

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
        product: Iterable[Tuple[Any, ...]] = itertools.product(*values)

        # Get subset of rows if partitioned
        if worker_num is not None and num_workers is not None:
            product = list(product)
            num_rows = len(product)
            # // is the floor (or integer) division operator
            start = worker_num * num_rows // num_workers
            stop = (worker_num + 1) * num_rows // num_workers
            product = product[start:stop]

        # Create a key-value mapping for each argument in each row
        matrix = []
        for row in product:
            params = {k: v for k, v in zip(keys, row)}
            if prune is not None and prune(**params):
                continue
            matrix.append(params)

        return matrix

    return _create


class Matrix:
    """
    A function to be invoked multiple times with parameters from rows of a
    cartesian product matrix of all possible parameters.

    This class is not intended to be used directly, but is used internally by
    the `matrix` decorator.
    """

    def __init__(self, func: Callable, **kwargs):
        self.func = func
        self.kwargs = kwargs
        self.worker_num: Optional[int] = None
        self.num_workers: Optional[int] = None
        self.pruner: Optional[Callable[..., bool]] = None

    @property
    def matrix(self):
        """The generated matrix of arguments"""
        return create_matrix(
            worker_num=self.worker_num,
            num_workers=self.num_workers,
            prune=self.pruner,
        )(**self.kwargs)

    def __call__(self, *args, **kwargs):
        """
        Invokes the function once for each row of the matrix. Any given keyword
        arguments are merged with the keyword arguments from the matrix row.

        Args:
            *args: Positional arguments to be included with each function
                invocation
            **kwargs: Keyword arguments to be merged with the matrix row
                arguments for each function invocation

        Returns:
            List of return values from each function invocation
        """
        results = []
        for it in self.matrix:
            it_args = deepcopy(args)
            it_kwargs = deepcopy(kwargs)
            it_kwargs.update(it)
            try:
                results.append(self.func(*it_args, **it_kwargs))
            except Exception as err:
                results.append(err)
        return results

    def override(self, **kwargs):
        """Overrides the input parameter constraints for the matrix."""
        self.kwargs.update(kwargs)
        return self

    def partition(self, worker_num: Optional[int], num_workers: Optional[int]):
        """
        Specifies the worker index and count used to partition the matrix for
        parallel-worker applications.
        """
        self.worker_num = worker_num
        self.num_workers = num_workers
        return self

    def prune(self, pruner: Optional[Callable[..., bool]]):
        """Specifies the pruner callable to be used when generating the matrix."""
        self.pruner = pruner
        return self

    def parallel(self, max_workers):
        """
        Creates a thread pool in which to perform the invoked function for each
        row of the matrix.

        Example::

            @matrix(x=[1, 2], y=[3, 4])
            def perform(x, y):
                return x * y

            perform.parallel(2)()  # Will use up to 2 threads

        Args:
            Maximum number of workers to use in the thread pool

        Returns:
            A function that will accept additional arguments to be forwarded to
            the invoked function and execute all rows of the matrix within the
            thread pool. The return of the function will be a list of all return
            values from each invocation of the matrix rows.
        """
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        def _run(*args, **kwargs):
            futures = []
            for it in self.matrix:
                it_args = deepcopy(args)
                it_kwargs = deepcopy(kwargs)
                it_kwargs.update(it)
                futures.append(executor.submit(self.func, *it_args, **it_kwargs))

            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as err:
                    results.append(err)
            return results

        return _run


def matrix(**kwargs):
    """
    Create a decorator to invoke the wrapped function once for each row of the
    parameter matrix created by the cartesian product of all input parameter
    values.

    Example::

        >>> from armory_matrix import matrix

        >>> @matrix(x=range(5))
        >>> def perform(a, x, b):
        >>>    return (a * x) + b

        >>> perform(2, b=3)
        [3, 5, 7, 9, 11]

        >>> # to filter by pruning...
        >>> perform.prune(lambda x: x % 2 == 1)(2, b=3)
        [3, 7, 11]
        >>> perform.prune(None) # to clear pruning

        >>> # to partition
        >>> perform.partition(0, 2)(2, b=3)
        [3, 5]
        >>> perform.partition(1, 2)(2, b=3)
        [7, 9, 11]
        >>> perform.parition(None, None) # to clear partition

        >>> # to override arguments
        >>> perform.override(x=range(3, 7))(2, b=3)
        [9, 11, 13, 15]

    Args:
        **kwargs: Keyword arguments to produce the parameter matrix. Argument
            values should be iterables (e.g., generator, list) of the allowable
            values for their respective named argument.

    Returns:
        Function decorator to wrap a given function in a callable `Matrix` object.
    """

    def decorator(func):
        return Matrix(func, **kwargs)

    return decorator
