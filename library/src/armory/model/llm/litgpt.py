"""Armory model wrapper for LitGPT models."""

from typing import TYPE_CHECKING, Optional

from armory.data import TextPromptBatch
from armory.evaluation import ModelProtocol
from armory.model.base import ArmoryModel

if TYPE_CHECKING:
    from litgpt import LLM


class LitGPT(ArmoryModel, ModelProtocol):
    """
    Wrapper around a LitGPT model that produces generated responses.
    """

    def __init__(
        self,
        name: str,
        model: "LLM",
        static_context: Optional[str] = None,
    ):
        """
        Initializes the model wrapper.

        :param name: Name of the model
        :type name: str
        :param model: LitGPT model being wrapped
        :type model: LLM
        """
        super().__init__(
            name=name,
            model=model,
        )
        self._static_context = static_context

    def predict(self, batch: TextPromptBatch):
        """
        Invokes the wrapped model using the text inputs in the given batch and
        updates the predictions in the batch.

        :param batch: Text prompt batch
        :type batch: TextPromptBatch
        """
        prompts = batch.inputs.get()
        contexts = batch.contexts.get()
        if contexts:
            prompts = [f"{c} {p}" for c, p in zip(contexts, prompts)]
        if self._static_context:
            prompts = [f"{self._static_context} {p}" for p in prompts]

        responses = []
        for prompt in prompts:
            response = self._model.generate(prompt)
            print("====")
            print(f"{prompt=}")
            print(f"{response=}")
            print("====")
            responses.append(response)

        batch.predictions.set(responses)

    def loss(self, batch: TextPromptBatch):
        raise NotImplementedError()
