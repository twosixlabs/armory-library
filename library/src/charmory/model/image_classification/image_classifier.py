from typing import Optional

from charmory.data import Batch, BatchedImages, TorchAccessor
from charmory.evaluation import ModelProtocol
from charmory.model.base import ArmoryModel, ModelInputAdapter, ModelOutputAdapter


class ImageClassifier(ArmoryModel, ModelProtocol):
    def __init__(
        self,
        name: str,
        model,
        accessor: BatchedImages.Accessor,
        preadapter: Optional[ModelInputAdapter] = None,
        postadapter: Optional[ModelOutputAdapter] = None,
    ):
        super().__init__(
            name=name,
            model=model,
            preadapter=preadapter,
            postadapter=postadapter,
        )
        self.accessor = accessor

    def _apply(self, *args, **kwargs):
        super()._apply(*args, **kwargs)
        if isinstance(self.accessor, TorchAccessor):
            self.accessor.to(device=self.device)

    def predict(self, batch: Batch):
        inputs = self.accessor.get(batch.inputs)
        outputs = self(inputs)
        batch.predictions.update(outputs)
