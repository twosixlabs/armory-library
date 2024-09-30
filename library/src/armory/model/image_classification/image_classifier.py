from typing import Optional

from armory.data import ImageClassificationBatch, ImageSpec, TorchSpec
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
            inputs_spec=armory.data.TorchImageSpec(
                dim=armory.data.ImageDimensions.CHW,
                scale=armory.data.Scale(
                    dtype=armory.data.DataType.FLOAT,
                    max=1.0,
                ),
            ),
        )
    """

    def __init__(
        self,
        name: str,
        model,
        inputs_spec: ImageSpec,
        preadapter: Optional[ModelInputAdapter] = None,
        postadapter: Optional[ModelOutputAdapter] = None,
    ):
        """
        Initializes the model wrapper.

        :param name: Name of the model
        :type name: str
        :param model: Image classification model being wrapped
        :type model: _type_
        :param inputs_spec: Data specification used to obtain raw image data from the
                image inputs contained in image classification batches.
        :type inputs_spec: ImageSpec
        :param preadapter: Optional, model input adapter, defaults to None
        :type preadapter: ModelInputAdapter, optional
        :param postadapter: Optional, model output adapter, defaults to None
        :type postadapter: ModelOutputAdapter, optional
        """
        super().__init__(
            name=name,
            model=model,
            preadapter=preadapter,
            postadapter=postadapter if postadapter is not None else self._postadapt,
        )
        self.inputs_spec = inputs_spec

    def _apply(self, *args, **kwargs):
        super()._apply(*args, **kwargs)
        if isinstance(self.inputs_spec, TorchSpec):
            self.inputs_spec.to(device=self.device)

    def _postadapt(self, output):
        if hasattr(output, "logits"):
            return output.logits
        if hasattr(output, "probs"):
            return output.probs
        if hasattr(output, "scores"):
            return output.scores
        return output

    def loss(self, batch: ImageClassificationBatch):
        raise NotImplementedError()

    def predict(self, batch: ImageClassificationBatch):
        """
        Invokes the wrapped model using the image inputs in the given batch and
        updates the image classification predictions in the batch.

        :param batch: Image classification batch
        :type batch: ImageClassificationBatch
        """
        inputs = batch.inputs.get(self.inputs_spec)
        outputs = self(inputs)
        batch.predictions.set(outputs)
