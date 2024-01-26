from typing import Optional

from armory.data import Batch, Images, TorchAccessor
from armory.evaluation import ModelProtocol
from armory.model.base import ArmoryModel, ModelInputAdapter, ModelOutputAdapter


class ImageClassifier(ArmoryModel, ModelProtocol):
    """
    Wrapper around a model that produces image classification predictions.

    This wrapper automatically applies a postadapter to the models outputs to
    extract the `logits`, `probs` or `scores` attribute from the returned object
    if any such attribute is found. Otherwise the unmodified model output is
    returned.

    Example::

        import armory.data
        from armory.model.image_classification import ImageClassifier

        # assuming `model` has been defined elsewhere
        classifier = ImageClassifier(
            name="My model",
            model=model,
            accessor=armory.data.Images.as_torch(
                dim=armory.data.ImageDimensions.CHW
            ),
        )
    """

    def __init__(
        self,
        name: str,
        model,
        accessor: Images.Accessor,
        preadapter: Optional[ModelInputAdapter] = None,
        postadapter: Optional[ModelOutputAdapter] = None,
    ):
        """
        Initializes the model wrapper.

        Args:
            name: Name of the model.
            model: Image classification model being wrapped.
            accessor: Data accessor used to obtain low-level image data from the
                highly-structured image inputs contained in image classification
                batches.
            preadapter: Optional, model input adapter.
            postadapter: Optional, model output adapter.
        """
        super().__init__(
            name=name,
            model=model,
            preadapter=preadapter,
            postadapter=postadapter if postadapter is not None else self._postadapt,
        )
        self.accessor = accessor

    def _apply(self, *args, **kwargs):
        super()._apply(*args, **kwargs)
        if isinstance(self.accessor, TorchAccessor):
            self.accessor.to(device=self.device)

    def _postadapt(self, output):
        if hasattr(output, "logits"):
            return output.logits
        if hasattr(output, "probs"):
            return output.probs
        if hasattr(output, "scores"):
            return output.scores
        return output

    def predict(self, batch: Batch):
        """
        Invokes the wrapped model using the image inputs in the given batch and
        updates the image classification predictions in the batch.

        Args:
            batch: Image classification batch
        """
        inputs = self.accessor.get(batch.inputs)
        outputs = self(inputs)
        batch.predictions.update(outputs)
