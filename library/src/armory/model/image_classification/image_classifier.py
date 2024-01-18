from typing import Optional

from armory.data import Batch, Images, TorchAccessor
from armory.evaluation import ModelProtocol
from armory.model.base import ArmoryModel, ModelInputAdapter, ModelOutputAdapter


class ImageClassifier(ArmoryModel, ModelProtocol):
    def __init__(
        self,
        name: str,
        model,
        accessor: Images.Accessor,
        preadapter: Optional[ModelInputAdapter] = None,
        postadapter: Optional[ModelOutputAdapter] = None,
    ):
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
        inputs = self.accessor.get(batch.inputs)
        outputs = self(inputs)
        batch.predictions.update(outputs)
