import time


class Engine:
    def __init__(self, evaluation):
        if not hasattr(evaluation, "eval_id"):
            evaluation.eval_id = str(time.time())

        # TODO: Refactor the dynamic import mechanism. -CW
        self.evaluation = evaluation
        self.scenario = evaluation.scenario.function(self.evaluation)

    def run(self):
        results = self.scenario.evaluate()
        return results
