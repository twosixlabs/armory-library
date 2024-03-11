"""Range generation utilities"""

from typing import Generator, Optional, overload


class frange:
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
    """

    @overload
    def __init__(self, stop: float, /): ...

    @overload
    def __init__(self, start: float, stop: float, step: Optional[float] = None): ...

    def __init__(
        self, start: float, stop: Optional[float] = None, step: Optional[float] = None
    ):
        """
        Initializes the floating point range.

        Args:
            start: The first value to be generated (or `0.0` if the parameter was
                not supplied)
            stop: The exclusive upper bound for values to be generated
            step: Difference between generated values (or `1.0` if the parameter was
                not supplied)
        """
        # If only start is given, then treat it as the stop and start from 0.0
        if stop is None:
            self.stop = float(start)
            self.start = 0.0
        else:
            self.stop = float(stop)
            self.start = float(start)

        self.step = float(step) if step is not None else 1.0
        if self.step == 0.0:
            raise ValueError()

    def __iter__(self) -> Generator[float, None, None]:
        """
        Creates iterable floating point generator.

        Returns:
            Generator of floating point numbers in the range `[start,stop)`
            incrementing by `step`
        """
        count = 0
        while True:
            temp = self.start + count * self.step
            if self.step > 0 and temp >= self.stop:
                break
            elif self.step < 0 and temp <= self.stop:
                break
            yield temp
            count += 1
