![armory-matrix logo](../docs/assets/armory-matrix-logo.png)

---

# Overview

Armory-matrix provides a utility to automatically expand parameter
specifications in order to evaluate all combinations of parameters.

For example, given two parameters `x` and `y` where `x` can be an integer
between 1 and 3 and `y` can be 1.5 or 6.5, we have the following matrix of
parameter combinations:

&nbsp; | x | y
-------|---|--
1      | 1 | 1.5
2      | 1 | 6.5
3      | 2 | 1.5
4      | 2 | 6.5
5      | 3 | 1.5
6      | 3 | 6.5

As the range of values for each parameter increases, and as more variable
parameters are added, the matrix becomes significantly larger.

# Installation

```bash
pip install armory-matrix
```

# Usage

Use the `matrix` decorator to specify a set or range of values for parameters of
a function. When the decorated function is invoked, it will automatically be
called once for each combination of input parameters.

```python
from armory.matrix import matrix

@matrix(
    x=range(1, 3),
    y=[1.5, 6.5],
)
def print_xy(x, y):
    print(f"{x=}, {y=}")

print_xy()
```

Will produce the following output:

```
x=1, y=1.5
x=1, y=6.5
x=2, y=1.5
x=2, y=6.5
```

This is roughly equivalent to the nested loops below:

```python
def print_xy(x, y):
    print(f"{x=}, {y=}")

for x in range(1, 3):
    for y in [1.5, 6.5]:
        print_xy(x=x, y=y)
```

## Parameter inspection

The resulting parameter matrix may be inspected via the `num_rows` and `matrix`
properties of the decorated function.

```python
assert print_xy.num_rows == 4

for row in print_xy.matrix:
    print(row)
```

Will produce the following output:

```
{'x': 1, 'y': 1.5}
{'x': 1, 'y': 6.5}
{'x': 2, 'y': 1.5}
{'x': 2, 'y': 6.5}
```

## Parameter range specification

The parameter ranges specified with the `@matrix` decorator can be any type as
long as it is iterable. When a non-iterable value is used, it is effectively a
fixed value for all rows of the matrix. The following are all valid matrix
parameter values:

```python
def gen():
    for d in [None, int, str]:
        yield d

@matrix(
    a=gen,                        # generator function
    b=(b for b in (True, False)), # generator
    c=42,                         # fixed value
    d=range(5),                   # iterable object (via __iter__)
    e=["odd", "even"],            # sequence (set, tuple, list)
)
def foo(a, b, c, d):
    pass
```

See the note below in [Dynamic parameters](#generator-functions) regarding use of
generator functions.

## Return values

The return value from calling the decorated function will be a sequence of all
the return values for each invocation of the function with parameters from rows
of the matrix. If the function raises an exception, the caught exception will be
the return value for that call.

```python
@matrix(a=range(1, 3), b=range(3))
def division(a, b):
    return a / b

print(division())
```

Will produce the following output:

```
[
    ZeroDivisionError('division by zero'), # 1 / 0
    1.0,                                   # 1 / 1
    0.5,                                   # 1 / 2
    ZeroDivisionError('division by zero'), # 2 / 0
    2.0,                                   # 2 / 1
    1.0,                                   # 2 / 2
]
```

## Partial parameter specification

It is also possible to only specify a subset of a function's arguments as matrix
parameters. Any arguments not defined with the `@matrix` decorator must be
specified when the function is invoked.

```python
@matrix(x=range(1, 4))
def quadratic(a, x, b):
    return (a * x) + b

assert quadratic(a=2, b=10) == [12, 14, 16]
```

## Post-definition overrides

Parameters may be overridden after definition:

```python
print_xy.override(x=[7])()
```

```
x=7, y=1.5
x=7, y=6.5
```

## Dynamic parameters

Dynamic or dependent parameters may be generated using a callable rather than a
value range.

```python
def y(x):
    if x == 1:
        return [1.5, 6.5]
    return [1.7, 2.3, 3.4]

@matrix(
    x=range(1, 3),
    y=y,
)
def print_xy(x, y):
    print(f"{x=}, {y=}")

print_xy()
```

```
x=1, y=1.5
x=1, y=6.5
x=2, y=1.7
x=2, y=2.3
x=2, y=3.4
```

This is roughly equivalent to the following:

```python
def y(x):
    if x == 1:
        return [1.5, 6.5]
    return [1.7, 2.3, 3.4]

def print_xy(x, y):
    print(f"{x=}, {y=}")

for x in range(1, 3):
    for _y in y(x=x):
        print_xy(x=x, y=_y)
```

The callable will be invoked with all prior generated parameters, in the order
they are defined in the `@matrix` decorator. It is possible to specify multiple
dynamic parameters, where the latter parameters depend on the former.

### Generator Functions

A custom generator function as a parameter range is effectively equivalent to a
dynamic parameter range. This is intentional, since generators cannot be rewound
and it is often necessary to iterate multiple times through a `matrix` parameter
space. However, this means that you may need to account for the prior parameter
arguments that will be passed to the generator function.

```python
def gen(**kwargs):
    for d in [None, int, str]:
        yield d

@matrix(
    a=range(5),
    b=gen,
)
def foo(a, b):
    pass
```

If we omitted the `**kwargs` from `gen`, we would encounter an "unexpected
keyword argument" error.

## Filtering

Rows may be omitted from execution by providing a filter callable:

```python
assert print_xy.num_rows == 4
filtered_print_xy = print_xy.filter(lambda x, y: x == 1 and y == 1.6)
assert filtered_print_xy.num_rows == 3
filtered_print_xy()
```

Any row for which the filter callable returns `True` will be omitted from the
matrix.

```
x=1, y=1.5
x=2, y=1.5
x=2, y=6.5
```

## Partitioning

For distributed runs (e.g., as a job submitted to a SLURM cluster), one may
specify a partition of the matrix to be run:

```python
print_xy[0::2]()  # invoke partition 1 of 2
print("---")
print_xy[1::2]()  # invoke partition 2 of 2
```

```
x=1, y=1.5
x=1, y=6.5
---
x=2, y=1.5
x=2, y=6.5
```

Typical Python slice operations apply, including use of a stop index. Note
however, that negative start or stop values are not supported.

```python
print_xy[10]() # Skip the first 10 rows
print_xy[5:10]() # Execute rows 5 through 9
```

## Parallelization

Function invocations may be parallelized within the Python process by using a
thread pool and specifying the max number of workers in the pool:

```python
print_xy.parallel(2)()
```
