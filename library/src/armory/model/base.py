"""Base Armory model wrapper."""

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from armory.track import Trackable

Args = Tuple[Any, ...]
Kwargs = Dict[str, Any]
ModelInputAdapter = Callable[..., Tuple[Args, Kwargs]]
"""
An adapter for model inputs. The output must be a tuple of args and kwargs
for the model's `forward` method.
"""

ModelOutputAdapter = Callable[[Any], Any]


class ArmoryModel(Trackable, nn.Module):
    """
    Wrapper around a model to apply an adapter to inputs and outputs of the
    model.

    Example:

        from armory.model import ArmoryModel

        def preadapter(images, *args, **kwargs):
            # Apply some transform to images
            return (images,) + args, kwargs

        def postadapter(output):
            # Apply some transform to output
            return output

        # assuming `model` has been defined elsewhere
        wrapper = ArmoryModel(
            "MyModel",
            model,
            preadapter=preadapter,
            postadapter=postadapter,
        )
    """

    def __init__(
        self,
        name: str,
        model,
        preadapter: Optional[ModelInputAdapter] = None,
        postadapter: Optional[ModelOutputAdapter] = None,
    ):
        """
        Initializes the model wrapper.

        :param name: Name of the model
        :type name: str
        :param model: Model being wrapped
        :type model: _type_
        :param preadapter: Optional, model input adapter, defaults to None
        :type preadapter: ModelInputAdapter, optional
        :param postadapter: Optional, model output adapter, defaults to None
        :type postadapter: ModelOutputAdapter, optional
        """
        super().__init__()
        self.name = name
        self._preadapter = preadapter
        self._model = model
        self._postadapter = postadapter
        self.device = torch.device("cpu")

    def _apply(self, fn, *args, **kwargs):
        self.device = fn(torch.zeros(1)).device
        return super()._apply(fn, *args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Applies pre- or postadapters, as appropriate and invokes the wrapped
        model
        """
        if self._preadapter:
            args, kwargs = self._preadapter(*args, **kwargs)

        output = self._model(*args, **kwargs)

        if self._postadapter:
            output = self._postadapter(output)

        return output
