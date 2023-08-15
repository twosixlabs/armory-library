"""
General image classification scenario
"""

from armory.instrument.export import ImageClassificationExporter
from charmory.scenario import Scenario
from charmory.scenarios.base import BaseScenario


class ImageClassificationTask(Scenario):
    def _load_sample_exporter(self):
        return ImageClassificationExporter(self.export_dir)


class ImageClassificationModule(BaseScenario):
    pass
