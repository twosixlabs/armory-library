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
    "worker_num,num_workers,expected",
    [
        (None, None, ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"]),
        (0, 1, ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"]),
        (1, 1, []),
        (0, 2, ["ad", "af", "be", "cd", "cf"]),
        (1, 2, ["ae", "bd", "bf", "ce"]),
        (0, 3, ["ad", "bd", "cd"]),
        (1, 3, ["ae", "be", "ce"]),
        (2, 3, ["af", "bf", "cf"]),
        (0, 4, ["ad", "be", "cf"]),
        (1, 4, ["ae", "bf"]),
        (2, 4, ["af", "cd"]),
        (3, 4, ["bd", "ce"]),
        (0, 5, ["ad", "bf"]),
        (1, 5, ["ae", "cd"]),
        (2, 5, ["af", "ce"]),
        (3, 5, ["bd", "cf"]),
        (4, 5, ["be"]),
        (0, 6, ["ad", "cd"]),
        (1, 6, ["ae", "ce"]),
        (2, 6, ["af", "cf"]),
        (3, 6, ["bd"]),
        (4, 6, ["be"]),
        (5, 6, ["bf"]),
    ],
)
def test_partition(worker_num, num_workers, expected):
    @matrix(x="abc", y="def")
    def concat(x, y):
        return f"{x}{y}"

    assert concat.partition(worker_num, num_workers)() == expected


def test_pruning():
    @matrix(x="ab", y="cd", z="ef")
    def concat(x, y, z):
        return f"{x}{y}{z}"

    assert concat.prune(lambda x, y, z: x == "a" and z == "e")() == [
        # "ace", # pruned
        "acf",
        # "ade", # pruned
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


def test_parallel_with_pruning():
    @matrix(x="ab", y="cd")
    def concat(x, y):
        return f"{x}{y}"

    assert set(concat.prune(lambda x, y: x == "a" and y == "d").parallel(3)()) == {
        "ac",
        # "ad", # pruned
        "bc",
        "bd",
    }


def test_parallel_with_partition():
    @matrix(x="abc", y="def")
    def concat(x, y):
        return f"{x}{y}"

    assert set(concat.partition(1, 3).parallel(2)()) == {"ae", "be", "ce"}
