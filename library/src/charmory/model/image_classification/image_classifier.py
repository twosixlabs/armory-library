from typing import Optional

from charmory.batch import BatchedImages, ImageClassificationBatch, NDimArray
from charmory.evaluation import ModelProtocol
from charmory.model.base import ArmoryModel, ModelInputAdapter, ModelOutputAdapter


class ImageClassifier(ArmoryModel, ModelProtocol):
    def __init__(
        self,
        name: str,
        model,
        inputs_accessor: BatchedImages.Accessor,
        preadapter: Optional[ModelInputAdapter] = None,
        postadapter: Optional[ModelOutputAdapter] = None,
    ):
        super().__init__(
            name=name,
            model=model,
            preadapter=preadapter,
            postadapter=postadapter,
        )
        self.inputs_accessor = inputs_accessor

    def predict(self, batch: ImageClassificationBatch):
        inputs = self.inputs_accessor.get(batch.inputs)
        outputs = self(inputs)
        batch.predictions = NDimArray(outputs)
