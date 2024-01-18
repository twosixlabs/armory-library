"""Internal typing utilities"""

from functools import wraps
import inspect
import logging
from typing import Any, Callable, TypeVar, Union, get_args, get_origin

import numpy as np
import torch

# This was only added to the builtin `typing` in Python 3.10,
# so we have to use `typing_extensions` for 3.8 support
from typing_extensions import ParamSpec

logger = logging.getLogger(__name__)
P = ParamSpec("P")
T = TypeVar("T")


_COERCIONS = {
    np.ndarray: {
        torch.Tensor: lambda x: x.cpu().numpy(),
        Any: np.asarray,
    },
    torch.Tensor: {
        np.ndarray: torch.from_numpy,
        Any: torch.as_tensor,
    },
}


def coerce(value, to_type):
    if to_type == inspect.Parameter.empty or isinstance(value, to_type):
        return value
    from_type = type(value)
    if to_type in _COERCIONS and from_type in _COERCIONS[to_type]:
        return _COERCIONS[to_type][from_type](value)
    if Any in _COERCIONS[to_type]:
        return _COERCIONS[to_type][Any](value)
    raise TypeError(f"No registered coercion method for {to_type} from {from_type}")


def register_coercion(from_type, to_type, coercion):
    if to_type not in _COERCIONS:
        _COERCIONS[to_type] = {}
    _COERCIONS[to_type][from_type] = coercion


def _unwrap(annotation):
    if get_origin(annotation) is Union:
        for t in get_args(annotation):
            if t is not None:
                yield t
    else:
        yield annotation


def _get_name(func):
    if hasattr(func, "__name__"):
        return func.__name__
    if hasattr(func, "__class__"):
        return func.__class__.__name__
    return repr(func)


def autocoerce(func: Callable[P, T]) -> Callable[..., T]:
    signature = inspect.signature(func)

    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        bound = signature.bind(*args, **kwargs)
        for name, orig_value in bound.arguments.items():
            if name in signature.parameters and orig_value is not None:
                param = signature.parameters[name]
                for to_type in _unwrap(param.annotation):
                    try:
                        bound.arguments[name] = coerce(orig_value, to_type)
                        break
                    except TypeError:
                        pass
                else:
                    raise TypeError(
                        f"Unable to coerce parameter, {name}, of type {type(orig_value)} "
                        f"to {param.annotation}"
                    )
                logger.warning(
                    "%s parameter %s coerced from %s to %s",
                    _get_name(func),
                    name,
                    type(orig_value),
                    type(bound.arguments[name]),
                )
        return func(*bound.args, **bound.kwargs)

    return wrapper
