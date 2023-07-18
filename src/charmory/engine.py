class Engine:
    def __init__(self, evaluation):
        self.evaluation = evaluation
        self.scenario = evaluation.scenario.function(self.evaluation)

    def run(self):
        results = self.scenario.evaluate()
        return results
