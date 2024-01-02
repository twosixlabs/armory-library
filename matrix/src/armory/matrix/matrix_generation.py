"""Matrix generation utilities"""

import concurrent.futures
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

# This was only added to the builtin `typing` in Python 3.10,
# so we have to use `typing_extensions` for 3.8 support
from typing_extensions import ParamSpec


def product(
    prior: Dict[str, Any], remaining: List[Tuple[str, Any]]
) -> Iterable[Mapping[str, Any]]:
    """
    Generates a cartesian product of parameters.

    This function works recursively, iterating over each parameter range in a
    depth-first manner. We do this rather than use `itertools.product` in order
    to support dynamic, or dependent, parameter ranges that are created based
    on the preceding parameter values.

    Args:
        prior: Key-value parameter mapping of parameters generated so far
        remaining: List of remaining key and value-iterable pairs

    Yields:
        Key-value parameter mappings for each row in the cartesian product
        matrix
    """
    if not remaining:
        yield prior
    else:
        key, values = remaining[0]
        if callable(values):
            values = values(**prior)
        if not isinstance(values, Iterable):
            values = [values]
        for value in values:
            next = deepcopy(prior)
            next[key] = value
            for params in product(next, remaining[1:]):
                yield params


def is_in_partition(index: int, partition: slice) -> bool:
    """Checks if the given row index is included in the partition, or slice"""
    if partition.start is not None and index < partition.start:
        return False
    if partition.stop is not None and index >= partition.stop:
        return False
    if partition.step is not None:
        return (index % partition.step) == (partition.start % partition.step)
    return True


def create_matrix(
    partition: Optional[slice] = None,
    filter: Optional[Callable[..., bool]] = None,
):
    """
    Creates a matrix of variable parameter values as a sequence of key-value
    argument mappings. The matrix is created as the cartesian product of all
    possible argument combinations.

    This function is not intended to be used directly, but is used internally
    by the `Matrix` class.

    Example::

        >>> from armory.matrix import create_matrix
        >>> list(create_matrix()(a=[1, 2], b=[3, 4]))
        [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}, {'a': 2, 'b': 4}]
        >>> list(create_matrix(slice(1, None, 2))(a=[1, 2], b=[3, 4]))
        [{'a': 1, 'b': 4}, {'a': 2, 'b': 4}]
        >>> list(create_matrix(None, lambda a, b: a==2 and b==4)(a=[1, 2], b=[3, 4]))
        [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}]
        >>> list(create_matrix()(a=[1, 2], b=lambda a: (3, 4) if a == 1 else (5, 6)))
        [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 5}, {'a': 2, 'b': 6}]

    Args:
        partition: Partition, or slice, of the matrix rows to be executed.
        filter: A callable to determine whether a row of the matrix should be
            omitted. If the callable returns `True` for a given set of
            parameters, that parameter set, or row, will be omitted from the
            returned matrix.

    Returns:
        A generator of key-value mappings. The generator function accepts
        keyword arguments of iterables, ranges, or sequences specifying the
        allowable values for each argument. The yielded value of the generator
        is a key-value mapping of arguments corresponding to a row of the
        matrix.

        If an argument to the generator is a callable, it will be called with
        the preceding keyword parameter values for the current row. The return
        of the callable is the value--or iterable--for the keyword, from which
        to generate additional rows in the matrix.
    """

    def _generate(**kwargs) -> Iterable[Mapping[str, Any]]:
        index = 0
        for params in product({}, list(kwargs.items())):
            # Skip over entries that are filtered out
            if filter is not None and filter(**params):
                continue
            # Skip over entries when partioning
            if partition is None or is_in_partition(index, partition):
                yield params
            index += 1

    return _generate


P = ParamSpec("P")
T = TypeVar("T")


class Matrix(Generic[P, T]):
    """
    A function to be invoked multiple times with parameters from rows of a
    cartesian product matrix of all possible parameters.

    This class is not intended to be used directly, but is used internally by
    the `matrix` decorator.
    """

    def __init__(
        self,
        func: Callable[P, T],
        kwargs: Dict,
        partition: Optional[slice] = None,
        filter: Optional[Callable[P, bool]] = None,
    ):
        self.func = func
        self.kwargs = kwargs
        self._partition = partition
        self._filter = filter

    @property
    def matrix(self):
        """The generated matrix of arguments"""
        return create_matrix(
            partition=self._partition,
            filter=self._filter,
        )(**self.kwargs)

    @property
    def num_rows(self):
        """Count of all rows in the matrix."""
        return sum(1 for _ in self.matrix)

    def __call__(self, *args, **kwargs) -> Sequence[Union[T, Exception]]:
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
        results: List[Union[T, Exception]] = []
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
        """Creates a modified matrix with overridden input parameter constraints."""
        new_kwargs = deepcopy(self.kwargs)
        new_kwargs.update(kwargs)
        return Matrix(
            self.func,
            kwargs=new_kwargs,
            partition=self._partition,
            filter=self._filter,
        )

    def __getitem__(self, partition: slice) -> "Matrix[P, T]":
        """Creates a modified matrix that executes a partition, or slice, of rows."""
        return Matrix(
            self.func,
            kwargs=deepcopy(self.kwargs),
            partition=partition,
            filter=self._filter,
        )

    def filter(self, filter: Optional[Callable[P, bool]]):
        """Creates a modified matrix that executes a filtered set of rows of the matrix."""
        return Matrix(
            self.func,
            kwargs=deepcopy(self.kwargs),
            partition=self._partition,
            filter=filter,
        )

    def parallel(self, max_workers, timeout: Optional[float] = None):
        """
        Creates a thread pool in which to perform the invoked function for each
        row of the matrix.

        Example::

            @matrix(x=[1, 2], y=[3, 4])
            def perform(x, y):
                return x * y

            perform.parallel(2)()  # Will use up to 2 threads

        Args:
            max_workers: Maximum number of workers to use in the thread pool
            timeout: Maximum number of seconds to wait. If None, then there is
                no limit on the wait time.

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
            for future in concurrent.futures.as_completed(futures, timeout):
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

        >>> from armory.matrix import matrix

        >>> @matrix(x=range(5))
        ... def perform(a, x, b):
        ...    return (a * x) + b

        >>> perform(2, b=3)
        [3, 5, 7, 9, 11]

        >>> # to filter...
        >>> perform.filter(lambda x: x % 2 == 1)(2, b=3)
        [3, 7, 11]

        >>> # to partition
        >>> perform[0::2](2, b=3)
        [3, 7, 11]
        >>> perform[1::2](2, b=3)
        [5, 9]

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

    def decorator(func: Callable[..., T]):
        return Matrix(func, kwargs=kwargs)

    return decorator
