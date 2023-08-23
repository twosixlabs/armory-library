"""Utilities to support experiment tracking within Armory."""

from functools import wraps
import os
from pathlib import Path
import sys
from typing import (
    Any,
    Callable,
    Mapping,
    Optional,
    Sequence,
    Set,
    TypeVar,
    Union,
    overload,
)

import mlflow
import mlflow.cli
import mlflow.server

# This was only added to the builtin `typing` in Python 3.10,
# so we have to use `typing_extensions` for 3.8 support
from typing_extensions import ParamSpec

P = ParamSpec("P")
T = TypeVar("T")
TRACKED_PARAMS_ATTR = "__tracked_params__"


def _has_tracked_params(target: object) -> bool:
    """
    Checks if the given target object has tracked parameters. That is, it was created by a
    function that was wrapped with `track_params` or a class wrapped with `track_init_params`
    """
    return hasattr(target, TRACKED_PARAMS_ATTR)


def _get_tracked_params(target: object) -> Mapping[str, Any]:
    """Get tracked parameters from the given target object."""
    return getattr(target, TRACKED_PARAMS_ATTR)


def _set_tracked_params(target: object, params: Mapping[str, Any]):
    """Set tracked parameters for the given target object"""
    setattr(target, TRACKED_PARAMS_ATTR, params)


@overload
def track_params(
    *,
    prefix: Optional[str] = None,
    ignore: Optional[Sequence[str]] = None,
    self_is_target: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    ...


@overload
def track_params(
    _func: Callable[P, T],
    *,
    prefix: Optional[str] = None,
    ignore: Optional[Sequence[str]] = None,
    self_is_target: bool = False,
) -> Callable[P, T]:
    ...


def track_params(
    _func: Optional[Callable] = None,
    *,
    prefix: Optional[str] = None,
    ignore: Optional[Sequence[str]] = None,
    self_is_target: bool = False,
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
        self_is_target: First argument to the decorated function (i.e., self)
            is the target to which the tracked parameters should be applied
            (default is the return from the decorated function)

    Returns:
        Decorated function if `_func` was provided, else a function decorator
    """

    def _decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            _prefix = prefix if prefix else func.__name__

            params: Mapping[str, Any] = {
                f"{_prefix}._func": f"{func.__module__}.{func.__qualname__}"
            }

            for key, val in kwargs.items():
                if ignore and key in ignore:
                    continue
                params[f"{_prefix}.{key}"] = val

            result = func(*args, **kwargs)
            _set_tracked_params(args[0] if self_is_target else result, params)
            return result

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
        cls.__init__ = track_params(prefix=_prefix, ignore=ignore, self_is_target=True)(
            cls.__init__
        )
        return cls

    if _cls is None:
        return _decorator
    else:
        return _decorator(_cls)


def get_tracked_params(target: object, recursive: bool = True) -> Mapping[str, Any]:
    params = {}
    if _has_tracked_params(target):
        params.update(_get_tracked_params(target))
    # if recursive:
    #     for attr in dir(target):
    #         child_params = get_tracked_params(getattr(target, attr, {}))
    #         count = 0
    #         for key, val in child_params.items():
    #             if count:
    #                 key = f"{key}.{count}"

    # # MLFlow does not allow duplicates, so check the active
    # # run and adjust the prefix if needed
    # count = 0
    # param = _prefix
    # while param in active_run.data.params:
    #     count += 1
    #     param = f"{_prefix}.{count}"
    # if count:
    #     _prefix = f"{_prefix}.{count}"
    # active_run.data.params[_prefix] = True

    return params


def track_params_from(
    target: object,
    prior_keys: Optional[Set[str]] = None,
    prior_targets: Optional[Set[object]] = None,
    max_recursion_depth: int = 4,
):
    if prior_keys is None:
        prior_keys = set()
    if prior_targets is None:
        prior_targets = set()

    if target in prior_targets:
        return
    prior_targets.add(target)

    print(f"checking for params from {target}")
    if _has_tracked_params(target):
        print(f"logging params from {target}")
        params = _get_tracked_params(target)
        # MLFlow does not allow duplicate parameters, so check against prior
        # logged keys and adjust the parameter keys with a count if needed
        count = 0
        for key, val in params.items():
            print(f"  has key {key}")
            if count:
                key = f"{key}.{count}"
            else:
                tmp = key
                while tmp in prior_keys:
                    count += 1
                    tmp = f"{key}.{count}"
                key = tmp
            print(f"  using key {key}")
            mlflow.log_param(key, val)
            prior_keys.add(key)

    if max_recursion_depth > 0:
        for attr in dir(target):
            child = getattr(target, attr, None)
            if child is not None:
                print(f"checking child {attr}")
                track_params_from(
                    child, prior_keys, prior_targets, max_recursion_depth - 1
                )


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
