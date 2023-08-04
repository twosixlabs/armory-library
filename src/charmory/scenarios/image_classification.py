"""
General image classification scenario
"""

from armory.instrument.export import ImageClassificationExporter
from charmory.scenario import Scenario


class ImageClassificationTask(Scenario):
    def _load_sample_exporter(self):
        return ImageClassificationExporter(self.export_dir)
