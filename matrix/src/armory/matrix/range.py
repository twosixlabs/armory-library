"""Range generation utilities"""

from typing import Generator, Optional, overload


@overload
def frange(stop: float, /) -> Generator[float, None, None]:
    ...


@overload
def frange(
    start: float, stop: float, step: Optional[float]
) -> Generator[float, None, None]:
    ...


def frange(
    start: float, stop: Optional[float] = None, step: Optional[float] = None
) -> Generator[float, None, None]:
    """
    A generator for a range of floating point numbers. If the `step` argument is
    omitted, it defaults to `1.0`. If the `start` argument is omitted, it
    defaults to `0`. If `step` is zero, `ValueError` is raised.

    For a positive `step`, the contents of a range `r` are determined by the
    formula `r[i] = start + step*i` where `i >= 0` and `r[i] < stop`.

    For a negative `step`, the contents of the range are still determined by the
    formula `r[i] = start + step*i`, but the constraints are `i >= 0` and `r[i] > stop`.

    Example::

        >>> from armory.matrix.range import frange
        >>> list(frange(5))
        [0.0, 1.0, 2.0, 3.0, 4.0]
        >>> list(frange(1, 6))
        [1.0, 2.0, 3.0, 4.0, 5.0]
        >>> list(frange(0, 3, 0.5))
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        >>> list(frange(0, -3, -0.5))
        [0.0, -0.5, -1.0, -1.5, -2.0, -2.5]

    Args:
        start: The first value to be generated (or `0.0` if the parameter was
            not supplied)
        stop: The exclusive upper bound for values to be generated
        step: Difference between generated values (or `1.0` if the parameter was
            not supplied)

    Returns:
        Generator of floating point numbers in the range `[start,stop)`
        incrementing by `step`
    """
    # If only start is given, then treat it as the stop and start from 0.0
    if stop is None:
        stop = start
        start = 0.0
    if step is None:
        step = 1.0
    if step == 0.0:
        raise ValueError()

    # Make sure all params are floats
    stop = float(stop)
    start = float(start)
    step = float(step)

    count = 0
    while True:
        temp = start + count * step
        if step > 0 and temp >= stop:
            break
        elif step < 0 and temp <= stop:
            break
        yield temp
        count += 1
