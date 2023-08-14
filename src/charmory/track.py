"""Utilities to support experiment tracking within Armory."""

from functools import wraps
import os
from pathlib import Path
import sys
from typing import Callable, Mapping, Optional, Sequence, TypeVar, Union, overload

import mlflow
import mlflow.cli
import mlflow.server

# This was only added to the builtin `typing` in Python 3.10,
# so we have to use `typing_extensions` for 3.8 support
from typing_extensions import ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


@overload
def track_params(
    *,
    prefix: Optional[str] = None,
    ignore: Optional[Sequence[str]] = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    ...


@overload
def track_params(
    _func: Callable[P, T],
    *,
    prefix: Optional[str] = None,
    ignore: Optional[Sequence[str]] = None,
) -> Callable[P, T]:
    ...


def track_params(
    _func: Optional[Callable] = None,
    *,
    prefix: Optional[str] = None,
    ignore: Optional[Sequence[str]] = None,
):
    """
    Create a decorator to log function keyword arguments as parameters with
    MLFlow.

    Example::

        from charmory.track import track_params

        @track_params()
        def load_model(name: str, batch_size: int):
            pass

        # Or for a third-party function that cannot have the decorator
        # already applied, you can apply it inline
        track_params(third_party_func)(arg=42)

    Args:
        _func: Optional function to be decorated
        prefix: Optional prefix for all keyword argument names (default is
            inferred from decorated function name)
        ignore: Optional list of keyword arguments to be ignored

    Returns:
        Decorated function if `_func` was provided, else a function decorator
    """

    def _decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            active_run = mlflow.active_run()
            if active_run:
                _prefix = prefix if prefix else func.__name__

                # MLFlow does not allow duplicates, so check the active
                # run and adjust the prefix if needed
                count = 0
                param = _prefix
                while param in active_run.data.params:
                    count += 1
                    param = f"{_prefix}.{count}"
                if count:
                    _prefix = f"{_prefix}.{count}"
                active_run.data.params[_prefix] = True

                mlflow.log_param(
                    f"{_prefix}._func", f"{func.__module__}.{func.__qualname__}"
                )

                for key, val in kwargs.items():
                    if ignore and key in ignore:
                        continue
                    mlflow.log_param(f"{_prefix}.{key}", val)

            return func(*args, **kwargs)

        return _wrapper

    if _func is None:
        return _decorator
    else:
        return _decorator(_func)


@overload
def track_init_params(
    *,
    prefix: Optional[str] = None,
    ignore: Optional[Sequence[str]] = None,
) -> Callable[[T], T]:
    ...


@overload
def track_init_params(
    _cls: T,
    *,
    prefix: Optional[str] = None,
    ignore: Optional[Sequence[str]] = None,
) -> T:
    ...


def track_init_params(
    _cls: Optional[object] = None,
    *,
    prefix: Optional[str] = None,
    ignore: Optional[Sequence[str]] = None,
):
    """
    Create a decorator to log class dunder-init keyword arguments as parameters
    with MLFlow.

    Example::

        from charmory.track import track_init_params

        @track_init_params()
        class MyDataset:
            def __init__(self, batch_size: int):
                pass

        # Or for a third-party class that cannot have the decorator
        # already applied, you can apply it inline
        obj = track_init_params(ThirdPartyClass)(arg=42)

    Args:
        _cls: Optional class to be decorated
        prefix: Optional prefix for all keyword argument names (default is
            inferred from decorated class name)
        ignore: Optional list of keyword arguments to be ignored

    Returns:
        Decorated class if `_cls` was provided, else a class decorator
    """

    def _decorator(cls: T) -> T:
        _prefix = prefix if prefix else getattr(cls, "__name__", "")
        cls.__init__ = track_params(prefix=_prefix, ignore=ignore)(cls.__init__)
        return cls

    if _cls is None:
        return _decorator
    else:
        return _decorator(_cls)


def track_evaluation(
    name: str, description: Optional[str] = None, uri: Optional[Union[str, Path]] = None
):
    """
    Create a context manager for tracking an evaluation run with MLFlow.

    Example::

        from charmory.track import track_evaluation

        with track_evaluation("my_experiment"):
            # Perform evaluation run

    Args:
        name: Experiment name (should be the same between runs)
        description: Optional description of the run
        uri: Optional MLFlow server URI, defaults to ~/.armory/mlruns
    """

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        if uri is None:
            uri = Path(Path.home(), ".armory/mlruns")
        mlflow.set_tracking_uri(uri)

    experiment = mlflow.get_experiment_by_name(name)
    if experiment:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(name)

    return mlflow.start_run(
        experiment_id=experiment_id,
        description=description,
    )


def track_metrics(metrics: Mapping[str, Sequence[float]]):
    """
    Log the given metrics with MLFlow.

    Args:
        metrics: Mapping of metrics names to values
    """
    if not mlflow.active_run():
        return

    for key, values in metrics.items():
        for value in values:
            mlflow.log_metric(key, value)


def server():
    """Start the MLFlow server"""
    args = sys.argv[1:]
    if "--backend-store-uri" not in sys.argv:
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            path = Path(Path.home(), ".armory/mlruns")
            args.extend(["--backend-store-uri", path.as_uri()])

    mlflow.cli.server(args)
