import threading
import time

from armory_matrix import matrix
import pytest


def test_matrix():
    @matrix(x=[1, 2], y=[3, 4])
    def multiply(x, y):
        return x * y

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


def test_override():
    @matrix(x=[1, 2], y=[3, 4])
    def multiply(x, y):
        return x * y

    assert multiply.override(x=5)() == [15, 20]


@pytest.mark.parametrize(
    "worker_num,num_workers,expected",
    [
        (0, 1, ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"]),
        (1, 1, []),
        (0, 2, ["ad", "ae", "af", "bd"]),
        (1, 2, ["be", "bf", "cd", "ce", "cf"]),
        (0, 3, ["ad", "ae", "af"]),
        (1, 3, ["bd", "be", "bf"]),
        (2, 3, ["cd", "ce", "cf"]),
        (0, 4, ["ad", "ae"]),
        (1, 4, ["af", "bd"]),
        (2, 4, ["be", "bf"]),
        (3, 4, ["cd", "ce", "cf"]),
        (0, 5, ["ad"]),
        (1, 5, ["ae", "af"]),
        (2, 5, ["bd", "be"]),
        (3, 5, ["bf", "cd"]),
        (4, 5, ["ce", "cf"]),
        (0, 6, ["ad"]),
        (1, 6, ["ae", "af"]),
        (2, 6, ["bd"]),
        (3, 6, ["be", "bf"]),
        (4, 6, ["cd"]),
        (5, 6, ["ce", "cf"]),
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

    assert set(concat.partition(1, 3).parallel(2)()) == {"bd", "be", "bf"}
