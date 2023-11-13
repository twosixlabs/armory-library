"""Armory model wrapper for JATIC object detection models."""
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import jatic_toolbox.protocols

from charmory.model.base import ArmoryModel, ModelInputAdapter


class JaticObjectDetectionModel(ArmoryModel):
    """
    Model wrapper with pre-applied output adapter for JATIC object detection
    models.

    Example::
        import jatic_toolbox
        from charmory.model.object_detection import JaticObjectDetectionModel

        model = jatic_toolbox.load_model(
            provider=PROVIDER,
            model_name=MODEL_NAME,
            task="object-detection",
        )
        model = JaticObjectDetectionModel(model)
    """

    def __init__(
        self,
        model: "jatic_toolbox.protocols.ObjectDetector",
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
