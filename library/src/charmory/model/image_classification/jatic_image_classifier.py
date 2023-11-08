"""Armory model wrapper for JATIC image classification models."""
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import jatic_toolbox.protocols

from charmory.model.base import ArmoryModel, ModelInputAdapter


class JaticImageClassificationModel(ArmoryModel):
    """
    Model wrapper with pre-applied output adapter for JATIC image classification
    models.

    Example::

        import jatic_toolbox
        from charmory.model.image_classification import JaticImageClassificationModel

        model = jatic_toolbox.load_model(
            provider=PROVIDER,
            model_name=MODEL_NAME,
            task="image-classification",
        )
        model = JaticImageClassificationModel(model)
    """

    def __init__(
        self,
        model: "jatic_toolbox.protocols.ImageClassifier",
        preadapter: Optional[ModelInputAdapter] = None,
    ):
        """
        Initializes the model wrapper.

        Args:
            model: model being wrapped
            preadapter: Optional, model input adapter
        """
        super().__init__(model, preadapter=preadapter, postadapter=self._adapt)

    def _adapt(self, output):
        if hasattr(output, "logits"):
            return output.logits
        if hasattr(output, "probs"):
            return output.probs
        if hasattr(output, "scores"):
            return output.scores
        return output
