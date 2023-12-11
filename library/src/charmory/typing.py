"""Internal typing utilities"""

from functools import wraps
import inspect
from typing import Any, Callable, TypeVar

import numpy as np
import torch

# This was only added to the builtin `typing` in Python 3.10,
# so we have to use `typing_extensions` for 3.8 support
from typing_extensions import ParamSpec

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


def autocoerce(func: Callable[P, T]) -> Callable[..., T]:
    signature = inspect.signature(func)

    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        bound = signature.bind(*args, **kwargs)
        for name, orig_value in bound.arguments.items():
            if name in signature.parameters:
                param = signature.parameters[name]
                bound.arguments[name] = coerce(orig_value, param.annotation)
        return func(*bound.args, **bound.kwargs)

    return wrapper
