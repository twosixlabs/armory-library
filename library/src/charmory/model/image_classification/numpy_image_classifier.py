from typing import Any, Dict, Optional

from charmory.batch import Batch, NDimArray
from charmory.evaluation import ModelProtocol


class NumpyImageClassifier(ModelProtocol):
    def __init__(self, name: str, model, inputs_kwargs: Optional[Dict[str, Any]]):
        self.name = name
        self.model = model
        self.inputs_kwargs = inputs_kwargs or {}

    def predict(self, batch: Batch):
        inputs = batch.inputs.numpy(**self.inputs_kwargs)
        results = self.model(inputs)
        batch.predictions = NDimArray(results)
