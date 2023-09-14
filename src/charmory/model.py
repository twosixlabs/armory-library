"""Armory model APIs"""

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import torch.nn as nn

if TYPE_CHECKING:
    import jatic_toolbox.protocols

Args = Tuple[Any, ...]
Kwargs = Dict[str, Any]
ModelInputAdapter = Callable[..., Tuple[Args, Kwargs]]
"""
An adapter for model inputs. The output must be a tuple of args and kwargs
for the model's `forward` method.
"""

ModelOutputAdapter = Callable[[Any], Any]


class ArmoryModel(nn.Module):
    """Wrapper around a model to apply an adapter to inputs and outputs of the model"""

    def __init__(
        self,
        model,
        preadapter: Optional[ModelInputAdapter] = None,
        postadapter: Optional[ModelOutputAdapter] = None,
    ):
        super().__init__()
        self._preadapter = preadapter
        self._model = model
        self._postadapter = postadapter

    def forward(self, *args, **kwargs):
        if self._preadapter:
            args, kwargs = self._preadapter(*args, **kwargs)

        output = self._model(*args, **kwargs)

        if self._postadapter:
            output = self._postadapter(output)

        return output


class JaticImageClassificationModel(ArmoryModel):
    """Model wrapper with pre-applied output adapter for JATIC image classification models"""

    def __init__(
        self,
        model: "jatic_toolbox.protocols.ImageClassifier",
        preadapter: Optional[ModelInputAdapter] = None,
    ):
        super().__init__(model, preadapter=preadapter, postadapter=self._adapt)

    def _adapt(self, output):
        if hasattr(output, "logits"):
            return output.logits
        if hasattr(output, "probs"):
            return output.probs
        if hasattr(output, "scores"):
            return output.scores
        return output


class JaticObjectDetectionModel(ArmoryModel):
    """Model wrapper with pre-applied output adapter for JATIC object detection models"""

    def __init__(
        self,
        model: "jatic_toolbox.protocols.ObjectDetector",
        preadapter: Optional[ModelInputAdapter] = None,
    ):
        super().__init__(model, preadapter=preadapter, postadapter=self._adapt)

    def _adapt(self, output):
        if hasattr(output, "boxes"):
            # JATIC wrappers turn the original model output (list of dicts per image):
            #
            #     [{"boxes": [...], ...}, {"boxes": [...], ...}, ...]
            #
            # into this (dict of lists for all images)
            #
            #     {"boxes": [[...], ...], ""}
            #
            # Since ART needs it to be a list of dicts again, we have to undo what the
            # JATIC wrapper did.
            if hasattr(output, "logits"):
                new_output = []
                for boxes, logits in zip(output.boxes, output.logits):
                    new_output.append(dict(boxes=boxes, logits=logits))
                return new_output
            if hasattr(output, "probs"):
                new_output = []
                for boxes, probs in zip(output.boxes, output.probs):
                    new_output.append(dict(boxes=boxes, probs=probs))
                return new_output
            if hasattr(output, "scores") and hasattr(output, "labels"):
                new_output = []
                for boxes, scores, labels in zip(
                    output.boxes, output.scores, output.labels
                ):
                    new_output.append(dict(boxes=boxes, scores=scores, labels=labels))
                return new_output
        return output
