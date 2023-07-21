from armory.logs import log
from charmory.evaluation import Evaluation
from charmory.utils import (
    apply_art_postprocessor_defense,
    apply_art_preprocessor_defense,
)


class Engine:
    def __init__(self, evaluation: Evaluation):
        self.evaluation = evaluation
        # Need to apply pre/post-processor defenses before the attack is instantiated
        self.apply_defense()
        self.scenario = evaluation.scenario.function(self.evaluation)

    def train(self, nb_epochs=1):
        """
        Train the evaluation model using the configured training dataset.

        Args:
            nb_epochs: Number of epochs with which to perform training
        """
        assert self.evaluation.dataset.train_dataset is not None, (
            "Requested to train the model but the evaluation dataset does not "
            "provide a train_dataset"
        )
        log.info(
            f"Fitting {self.evaluation.model.name} model with "
            f"{self.evaluation.dataset.name} dataset..."
        )
        # TODO trainer defense when poisoning attacks are supported
        self.evaluation.model.model.fit_generator(
            self.evaluation.dataset.train_dataset,
            nb_epochs=nb_epochs,
        )

    def apply_defense(self):
        """Apply the evaluation defense, if any, to the evaluation model"""
        if self.evaluation.defense is not None:
            if self.evaluation.defense.type == "Preprocessor":
                apply_art_preprocessor_defense(
                    self.evaluation.model.model, self.evaluation.defense.defense
                )
            elif self.evaluation.defense.type == "Postprocessor":
                apply_art_postprocessor_defense(
                    self.evaluation.model.model, self.evaluation.defense.defense
                )

    def run(self):
        results = self.scenario.evaluate()
        return results
