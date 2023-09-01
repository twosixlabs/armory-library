"""
Sample exporting utilities
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from PIL.Image import Image
from mlflow.client import MlflowClient
import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure
    import pandas
    import plotly.graph_objects


class Exporter:
    """No-op exporter"""

    def log_image(self, image: Union[np.ndarray, Image], artifact_path: str):
        pass

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        pass

    def log_text(self, text: str, artifact_file: str):
        pass

    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str):
        pass

    def log_figure(
        self,
        figure: Union["matplotlib.figure.Figure", "plotly.graph_objects.Figure"],
        artifact_file: str,
    ):
        pass

    def log_table(
        self,
        data: Union[Dict[str, Any], "pandas.DataFrame"],
        artifact_file: str,
    ):
        pass


class MlflowExporter(Exporter):
    """
    Convenience wrapper around an MLFlow client to automatically invoke the
    client APIs with the run ID
    """

    def __init__(self, client: MlflowClient, run_id: str):
        self.client = client
        self.run_id = run_id

    def log_image(self, image: Union[np.ndarray, Image], artifact_path: str):
        # Make sure image data has channel last
        if (
            isinstance(image, np.ndarray)
            and len(image.shape) == 3
            and image.shape[0] in (1, 3, 4)
        ):
            image = np.transpose(image, (1, 2, 0))
        self.client.log_image(self.run_id, image, artifact_path)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        self.client.log_artifact(self.run_id, local_path, artifact_path)

    def log_text(self, text: str, artifact_file: str):
        self.client.log_text(self.run_id, text, artifact_file)

    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str):
        self.client.log_dict(self.run_id, dictionary, artifact_file)

    def log_figure(
        self,
        figure: Union["matplotlib.figure.Figure", "plotly.graph_objects.Figure"],
        artifact_file: str,
    ):
        self.client.log_figure(self.run_id, figure, artifact_file)

    def log_table(
        self,
        data: Union[Dict[str, Any], "pandas.DataFrame"],
        artifact_file: str,
    ):
        self.client.log_table(self.run_id, data, artifact_file)
