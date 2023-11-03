"""Armory model wrapper for JATIC image classification models."""
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import jatic_toolbox.protocols

from charmory.model.base import ArmoryModel, ModelInputAdapter


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
