"""Utilities to support experiment tracking within Armory."""

from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
import os
from pathlib import Path
import sys
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
    overload,
)

import mlflow
import mlflow.cli
import mlflow.server
import torch

# This was only added to the builtin `typing` in Python 3.10,
# so we have to use `typing_extensions` for 3.8 support
from typing_extensions import ParamSpec

from armory.logs import log

# Params are recorded globally in a stack of parameter stores, where the
# first stack entry is the default, implicit parameter store. Creation of
# subsequent parameter stores only occurs by using the `tracking_context`
# context manager. Params are ways recorded in the top-most store in the
# stack at the time of recording. This is modeled after the way MLFlow handles
# nested calls to `start_run`.
_params_stack: List[Dict[str, Any]] = []


def get_current_params() -> Dict[str, Any]:
    """Get the parameters from the current tracking context"""
    if len(_params_stack) == 0:
        _params_stack.append({})
    return _params_stack[-1]


def track_param(key: str, value: Any):
    """
    Record a parameter in the current tracking context to be logged with MLFlow.

    Example::

        from charmory.track import track_param

        track_param("key", "value")

    Args:
        key: Parameter name (should be unique or will overwrite previous values)
        value: Parameter value
    """
    params = get_current_params()
    if key in params:
        log.warning(
            f"Parameter {key} has already been logged with value {params[key]}, "
            f"and will be overwritten with value {value}. Use a unique parameter "
            "key argument or start a new tracking context with `tracking_context` "
            "to avoid this warning."
        )
    params[key] = value


def reset_params():
    """Clear all parameters in the current tracking context"""
    params = get_current_params()
    params.clear()


@contextmanager
def tracking_context(nested: bool = False):
    """
    Create a new tracking context. Parameters recorded while the context is
    active will be isolated from other contexts. Upon completion of the context,
    all parameters will be cleared.

    Example::

        from charmory.track import tracking_context, track_param

        with tracking_context():
            track_param("key", "value1")

        with tracking_context():
            track_param("key", "value2")

    Args:
        nested: Copy parameters from the current context into the new
            tracking context

    Returns:
        Context manager
    """
    new_context = {}
    if nested:
        new_context = deepcopy(get_current_params())
    _params_stack.append(new_context)
    try:
        yield
    finally:
        _params_stack.pop()


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
    Create a decorator to record function keyword arguments as parameters in the
    current tracking context to be logged with MLFlow.

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
            _prefix = prefix if prefix else func.__name__

            params = get_current_params()

            if f"{_prefix}._func" in params:
                log.warning(
                    f"Parameters with prefix {_prefix} have already been logged and will "
                    "be overwritten. Use a unique prefix or start a new tracking context "
                    "with `tracking_context` to avoid this warning."
                )
                # Remove prior params with this prefix
                for key in list(params.keys()):
                    if key.startswith(f"{_prefix}."):
                        params.pop(key)

            params[f"{_prefix}._func"] = f"{func.__module__}.{func.__qualname__}"

            for key, val in kwargs.items():
                if ignore and key in ignore:
                    continue
                params[f"{_prefix}.{key}"] = val

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
    Create a decorator to record class dunder-init keyword arguments as
    parameters in the current tracking context to be logged with MLFlow.

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
        cls.__init__ = track_params(prefix=_prefix, ignore=ignore)(cls.__init__)  # type: ignore
        return cls

    if _cls is None:
        return _decorator
    else:
        return _decorator(_cls)


def init_tracking_uri(armory_home: Path) -> str:
    """
    Initialize the MLFlow tracking URI.

    If the MLFLOW_TRACKING_URI environment variable is set, no change is made
    to the default tracking URI. Otherwise, the `mlruns` directory under the
    given armory home path will be set as the tracking URI.

    Args:
        armory_home: Path to armory home directory

    Return:
        Current MLFlow tracking URI
    """
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        uri = armory_home / "mlruns"
        mlflow.set_tracking_uri(uri)
    return mlflow.get_tracking_uri()


def track_evaluation(
    name: str, description: Optional[str] = None, uri: Optional[Union[str, Path]] = None
):
    """
    Create a context manager for tracking an evaluation run with MLFlow.
    Parameters that have been recorded in the current tracking context will be
    logged with MLFlow.

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

    run = mlflow.start_run(
        experiment_id=experiment_id,
        description=description,
    )

    mlflow.log_params(get_current_params())

    return run


def track_metrics(metrics: Mapping[str, Union[float, Sequence[float], torch.Tensor]]):
    """
    Log the given metrics with MLFlow.

    Args:
        metrics: Mapping of metrics names to values
    """
    if not mlflow.active_run():
        return

    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            mlflow.log_metric(key, value.item())
        elif isinstance(value, float):
            mlflow.log_metric(key, value)
        else:
            for val in value:
                mlflow.log_metric(key, val)


@contextmanager
def track_system_metrics(run_id: str):
    """
    Create a context in which to track system metrics and log them to the given
    MLflow experiment run. System metrics include CPU, disk, and network utilization
    metrics. If the `pynvml` package is installed, then GPU utilization metrics will
    also be collected.

    Example::

        from charmory.track import track_system_metrics
        import mlflow

        with mlflow.start_run() as active_run:
            with track_system_metrics(active_run.info.run_id)
                # Do something

    Args:
        run_id: MLflow experiment run ID of the run to which to record the
            system metrics

    Returns:
        Context manager
    """
    monitor = None
    try:
        from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor

        monitor = SystemMetricsMonitor(run_id)
        monitor.start()
    except Exception as err:
        log.warning(
            f"Exception creating system metrics monitor, {err}, system metrics will be unavailable for this run"
        )

    try:
        yield
    finally:
        if monitor is not None:
            try:
                monitor.finish()
            except Exception as err:
                log.warning(f"Exception shutting down system metrics monitor, {err}")


def server():
    """Start the MLFlow server"""
    args = sys.argv[1:]
    if "--backend-store-uri" not in sys.argv:
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            path = Path(Path.home(), ".armory/mlruns")
            args.extend(["--backend-store-uri", path.as_uri()])

    mlflow.cli.server(args)
