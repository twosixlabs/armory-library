"""Utilities to support experiment tracking within Armory."""

from functools import wraps
import os
from pathlib import Path
from typing import Callable, List, Optional, TypeVar, Union

import mlflow

# This was only added to the builtin `typing` in Python 3.10,
# so we have to use `typing_extensions` for 3.8 support
from typing_extensions import ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


def track_params(prefix: str, ignore: Optional[List[str]] = None):
    """
    Create a decorator to log function keyword arguments as parameters with
    MLFlow.

    Example::

        from charmory.track import track_params

        @track_params("model")
        def load_model(name: str, batch_size: int):
            pass

        # Or for a third-party function that cannot have the decorator
        # already applied, you can apply it inline
        track_params("third_party")(third_party_func)(arg=42)

    Args:
        prefix: String to be prefixed to all keyword argument names
        ignore: Optional list of keyword arguments to be ignored

    Returns:
        Function decorator
    """

    def _decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            mlflow.log_param(
                f"{prefix}._func", f"{func.__module__}.{func.__qualname__}"
            )
            for key, val in kwargs.items():
                if ignore and key in ignore:
                    continue
                mlflow.log_param(f"{prefix}.{key}", val)

            return func(*args, **kwargs)

        return _wrapper

    return _decorator


def track_init_params(prefix: str, ignore: Optional[List[str]] = None):
    """
    Create a decorator to log class dunder-init keyword arguments as parameters
    with MLFlow.

    Example::

        from charmory.track import track_init_params

        @track_init_params("dataset")
        class MyDataset:
            def __init__(self, batch_size: int):
                pass

        # Or for a third-party class that cannot have the decorator
        # already applied, you can apply it inline
        obj = track_init_params("third_party")(ThirdPartyClass)(arg=42)

    Args:
        prefix: String to be prefixed to all keyword argument names
        ignore: Optional list of keyword arguments to be ignored

    Returns:
        Class decorator
    """

    def _decorator(cls: T) -> T:
        cls.__init__ = track_params(prefix, ignore)(cls.__init__)
        return cls

    return _decorator


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
