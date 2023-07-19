class Engine:
    def __init__(self, evaluation):
        self.evaluation = evaluation
        self.scenario = evaluation.scenario.function(self.evaluation)

    def train(self, **fit_kwargs):
        """Train the model using the configured training dataset"""
        assert self.evaluation.dataset.train_dataset is not None, (
            "Requested to train the model but the evaluation dataset does not "
            "provide a train_dataset"
        )
        self.scenario.fit(**fit_kwargs)

    def run(self):
        results = self.scenario.evaluate()
        return results
