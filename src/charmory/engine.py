from importlib import import_module
import time


class Engine:
    def __init__(self, evaluation):
        self.evaluation = evaluation
        # TODO: Remove after refactor. -CW
        if not hasattr(self.evaluation, "eval_id"):
            self.evaluation.eval_id = str(time.time())

    def run(self):
        # TODO: Refactor the dynamic import mechanism. -CW
        _config = self.evaluation.scenario
        scenario_module, scenario_method = _config.function.split(":")
        ScenarioClass = getattr(import_module(scenario_module), scenario_method)

        return ScenarioClass(self.evaluation).evaluate()
