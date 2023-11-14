import doctest

import pytest

from armory.matrix import matrix
import armory.matrix.range
from armory.matrix.range import frange

pytestmark = pytest.mark.unit


def test_docstrings():
    results = doctest.testmod(armory.matrix.range)
    assert results.attempted > 0
    assert results.failed == 0


@pytest.mark.parametrize(
    "start,stop,step,expected",
    [
        (5, None, None, [0.0, 1.0, 2.0, 3.0, 4.0]),
        (5, None, 0.5, [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]),
        (2, 5, None, [2.0, 3.0, 4.0]),
        (2, 5, 0.5, [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]),
        (5, 2, -0.5, [5.0, 4.5, 4.0, 3.5, 3.0, 2.5]),
        (1.5, 2.0, 0.1, [1.5, 1.6, 1.7, 1.8, 1.9]),
    ],
)
def test_frange(start, stop, step, expected):
    assert list(frange(start, stop, step)) == expected


def test_frange_in_matrix():
    @matrix(
        x=frange(2, 4),
        y=frange(4, 2, -1.0),
    )
    def multiply(x, y):
        return x * y

    assert list(multiply.matrix) == [
        dict(x=2.0, y=4.0),
        dict(x=2.0, y=3.0),
        dict(x=3.0, y=4.0),
        dict(x=3.0, y=3.0),
    ]
