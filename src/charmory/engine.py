from importlib import import_module
import time


class Engine:
    def __init__(self, evaluation):
        if not hasattr(evaluation, "eval_id"):
            evaluation.eval_id = str(time.time())

        # TODO: Refactor the dynamic import mechanism. -CW
        scenario_module, scenario_method = evaluation.scenario.function.split(":")
        ScenarioClass = getattr(import_module(scenario_module), scenario_method)

        self.evaluation = evaluation
        self.model = self.evaluation.model
        self.dataset = self.evaluation.dataset
        self.attack = self.evaluation.attack
        self.scenario = ScenarioClass(self.evaluation)

    def run(self):
        results = self.scenario.evaluate()

        self.dataset = self.scenario.dataset
        self.model = self.scenario._loaded_model

        return results
