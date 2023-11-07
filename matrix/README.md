![charmory logo](../docs/assets/charmory.png)

---

# Overview

Armory-matrix provides a utility to automatically expand parameter
specifications in order to perform a series of evaluations covering all possible
combinations of parameters.

# Installation

```bash
pip install armory-matrix
```

# Usage

Use the `matrix` decorator to specify the ranges of values for all parameters of
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

Parameters may be overridden after definition:

```python
print_xy.override(x=[7])
print_xy()
```

```
x=7, y=1.5
x=7, y=6.5
```

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

For distributed runs (e.g., as a job submitted to a SLURM cluster), one may
specify a partition of the matrix to be run:

```python
print_xy.partition(0, 2)  # invoke partition 1 of 2
print_xy()
print("---")
print_xy.partition(1, 2)  # invoke partition 2 of 2
print_xy()
```

```
x=1, y=1.5
x=1, y=6.5
---
x=2, y=1.5
x=2, y=6.5
```

Function invocations may be parallelized using a thread pool by specifying the
max number of workers in the pool:

```python
print_xy.parallel(2)
print_xy()
```
