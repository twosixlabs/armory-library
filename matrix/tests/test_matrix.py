import doctest
import threading
import time

import pytest

from armory.matrix import matrix
import armory.matrix.matrix_generation

pytestmark = pytest.mark.unit


def test_docstrings():
    results = doctest.testmod(armory.matrix.matrix_generation)
    assert results.attempted > 0
    assert results.failed == 0


def test_matrix():
    @matrix(x=[1, 2], y=[3, 4])
    def multiply(x, y):
        return x * y

    assert multiply.num_rows == 4
    assert list(multiply.matrix) == [
        {"x": 1, "y": 3},
        {"x": 1, "y": 4},
        {"x": 2, "y": 3},
        {"x": 2, "y": 4},
    ]
    assert multiply() == [3, 4, 6, 8]


def test_single_value():
    @matrix(x=2, y=range(1, 7, 2))
    def multiply(x, y):
        return x * y

    assert multiply() == [2, 6, 10]


def test_fixed_arguments():
    @matrix(x=range(1, 4))
    def quadratic(a, x, b):
        return (a * x) + b

    assert all([isinstance(x, TypeError) for x in quadratic()])
    assert quadratic(2, b=10) == [12, 14, 16]


def test_exceptions():
    err = ValueError()

    @matrix(x=range(5))
    def raise_if_even(x):
        if x % 2 == 0:
            raise err
        return x

    assert raise_if_even() == [err, 1, err, 3, err]


def test_override():
    @matrix(x=[1, 2], y=[3, 4])
    def multiply(x, y):
        return x * y

    assert multiply.override(x=5)() == [15, 20]


@pytest.mark.parametrize(
    "start,stop,step,expected",
    [
        # Steps
        (None, None, None, ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"]),
        (0, None, 1, ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"]),
        (0, None, 2, ["ad", "af", "be", "cd", "cf"]),
        (1, None, 2, ["ae", "bd", "bf", "ce"]),
        (0, None, 3, ["ad", "bd", "cd"]),
        (1, None, 3, ["ae", "be", "ce"]),
        (2, None, 3, ["af", "bf", "cf"]),
        (0, None, 4, ["ad", "be", "cf"]),
        (1, None, 4, ["ae", "bf"]),
        (2, None, 4, ["af", "cd"]),
        (3, None, 4, ["bd", "ce"]),
        (0, None, 5, ["ad", "bf"]),
        (1, None, 5, ["ae", "cd"]),
        (2, None, 5, ["af", "ce"]),
        (3, None, 5, ["bd", "cf"]),
        (4, None, 5, ["be"]),
        (0, None, 6, ["ad", "cd"]),
        (1, None, 6, ["ae", "ce"]),
        (2, None, 6, ["af", "cf"]),
        (3, None, 6, ["bd"]),
        (4, None, 6, ["be"]),
        (5, None, 6, ["bf"]),
        # None-zero start
        (4, None, 2, ["be", "cd", "cf"]),
        (5, None, 2, ["bf", "ce"]),
        # Stops
        (0, 6, 2, ["ad", "af", "be"]),
        (1, 6, 2, ["ae", "bd", "bf"]),
        # Start and stop
        (2, 6, 2, ["af", "be"]),
        (3, 6, 2, ["bd", "bf"]),
        (2, 6, None, ["af", "bd", "be", "bf"]),
    ],
)
def test_slice(start, stop, step, expected):
    @matrix(x="abc", y="def")
    def concat(x, y):
        return f"{x}{y}"

    assert concat[start:stop:step]() == expected


def test_filtering():
    @matrix(x="ab", y="cd", z="ef")
    def concat(x, y, z):
        return f"{x}{y}{z}"

    assert concat.filter(lambda x, y, z: x == "a" and z == "e")() == [
        # "ace", # filtered
        "acf",
        # "ade", # filtered
        "adf",
        "bce",
        "bcf",
        "bde",
        "bdf",
    ]


def test_single_dynamic():
    def dynamic_z(x, y):
        if x == "b":
            return "ef"
        return "0"

    @matrix(x="ab", y="cd", z=dynamic_z)
    def concat(x, y, z):
        return f"{x}{y}{z}"

    assert concat() == [
        "ac0",
        "ad0",
        "bce",
        "bcf",
        "bde",
        "bdf",
    ]


def test_multiple_dynamic():
    def dynamic_y(x):
        if x == "b":
            return "cd"
        return "0"

    def dynamic_z(x, y):
        if y == "d":
            return "ef"
        return "0"

    @matrix(x="ab", y=dynamic_y, z=dynamic_z)
    def concat(x, y, z):
        return f"{x}{y}{z}"

    assert concat() == [
        "a00",
        "bc0",
        "bde",
        "bdf",
    ]


def test_parallel():
    @matrix(x="abc", y="def")
    def get_thread_id(x, y):
        time.sleep(0.01)
        return threading.get_ident()

    this_thread = threading.get_ident()

    thread_ids = set(get_thread_id.parallel(2)())
    assert this_thread not in thread_ids
    assert len(thread_ids) == 2


def test_parallel_with_fixed_arguments():
    @matrix(x=range(1, 4))
    def quadratic(a, x, b):
        return (a * x) + b

    assert set(quadratic.parallel(2)(2, b=10)) == {12, 14, 16}


def test_parallel_with_exceptions():
    err = ValueError()

    @matrix(x=range(5))
    def raise_if_even(x):
        if x % 2 == 0:
            raise err
        return x

    assert set(raise_if_even()) == {err, 1, 3}


def test_parallel_with_override():
    @matrix(x=[1, 2], y=[3, 4])
    def multiply(x, y):
        return x * y

    assert set(multiply.override(x=5).parallel(2)()) == {15, 20}


def test_parallel_with_filtering():
    @matrix(x="ab", y="cd")
    def concat(x, y):
        return f"{x}{y}"

    assert set(concat.filter(lambda x, y: x == "a" and y == "d").parallel(3)()) == {
        "ac",
        # "ad", # filtered
        "bc",
        "bd",
    }


def test_parallel_with_partition():
    @matrix(x="abc", y="def")
    def concat(x, y):
        return f"{x}{y}"

    assert set(concat[1::3].parallel(2)()) == {"ae", "be", "ce"}
