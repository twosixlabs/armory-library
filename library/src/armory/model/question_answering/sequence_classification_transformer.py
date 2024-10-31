"""Armory model wrapper for HuggingFace transformer models."""

from typing import TYPE_CHECKING

import torch

from armory.data import TextClassificationBatch
from armory.evaluation import ModelProtocol
from armory.model.base import ArmoryModel

if TYPE_CHECKING:
    from transformers import AutoModelForQuestionAnswering, PreTrainedTokenizer


class SequenceClassificationTransformer(ArmoryModel, ModelProtocol):
    """
    Wrapper around a HuggingFace transformer model that produces sequence
    classification predictions.
    """

    def __init__(
        self,
        name: str,
        model: "AutoModelForQuestionAnswering",
        tokenizer: "PreTrainedTokenizer",
    ):
        """
        Initializes the model wrapper.

        :param name: Name of the model
        :type name: str
        :param model: Sequence classification transformer being wrapped
        :type model: AutoModelForQuestionAnswering
        :param tokenizer: Transformer tokenizer for the model
        :type tokenizer: AutoTokenizer
        """
        super().__init__(
            name=name,
            model=model,
        )
        self.tokenizer = tokenizer

    def predict(self, batch: TextClassificationBatch):
        """
        Invokes the wrapped model using the text inputs in the given batch and
        updates the sequence classification predictions in the batch.

        :param batch: Text prompt batch
        :type batch: TextClassificationBatch
        """
        questions = batch.inputs.get()
        contexts = batch.contexts.get()
        if contexts:
            inputs = [e for e in zip(questions, contexts)]
        else:
            # Don't know if this is right, need to test it on a dataset without
            # contexts
            inputs = [(q, "") for q in questions]

        encoded_input = self.tokenizer(
            inputs, padding=True, truncation=True, return_tensors="pt"
        )
        outputs = self._model(**encoded_input)
        probabilities = torch.softmax(outputs.logits, dim=-1).cpu()
        batch.predictions.set(probabilities)

    def loss(self, batch: TextClassificationBatch):
        raise NotImplementedError()
