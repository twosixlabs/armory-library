class Engine:
    def __init__(self, evaluation):
        self.evaluation = evaluation
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
        self.scenario.fit(nb_epochs=nb_epochs)

    def run(self):
        results = self.scenario.evaluate()
        return results
