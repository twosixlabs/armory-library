"""
Sample export sinks/destinations
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from mlflow.client import MlflowClient
import numpy as np
import torch

if TYPE_CHECKING:
    import PIL.Image
    import matplotlib.figure
    import pandas
    import plotly.graph_objects


class Sink:
    """No-op export sink"""

    def log_image(
        self, image: Union[np.ndarray, "PIL.Image.Image"], artifact_path: str
    ):
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


class MlflowSink(Sink):
    """
    Convenience wrapper around an MLFlow client to automatically invoke the
    client APIs with the run ID
    """

    def __init__(self, client: MlflowClient, run_id: str):
        self.client = client
        self.run_id = run_id

    def log_image(
        self, image: Union[np.ndarray, "PIL.Image.Image"], artifact_path: str
    ):
        self.client.log_image(self.run_id, image, artifact_path)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        self.client.log_artifact(self.run_id, local_path, artifact_path)

    def log_text(self, text: str, artifact_file: str):
        self.client.log_text(self.run_id, text, artifact_file)

    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str):
        self.client.log_dict(self.run_id, _serialize(dictionary), artifact_file)

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


def _serialize(obj):
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return [_serialize(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _serialize(obj.tolist())
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj
