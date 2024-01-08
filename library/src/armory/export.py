"""
Sample exporting utilities
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from mlflow.client import MlflowClient
import numpy as np
import numpy.typing as npt
import torch
from torchvision.utils import draw_bounding_boxes

if TYPE_CHECKING:
    import PIL.Image
    import matplotlib.figure
    import pandas
    import plotly.graph_objects


class Exporter:
    """No-op exporter"""

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


class MlflowExporter(Exporter):
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
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_serialize(i) for i in obj.tolist()]
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


def draw_boxes_on_image(
    image: npt.NDArray[np.number],
    ground_truth_boxes: Optional[np.ndarray] = None,
    ground_truth_color: str = "red",
    ground_truth_width: int = 2,
    pred_boxes: Optional[np.ndarray] = None,
    pred_color: str = "white",
    pred_width: int = 2,
) -> npt.NDArray[np.uint8]:
    """
    Draw bounding boxes for ground truth objects and predicted objects on top of
    an image sample.

    Ground truth bounding boxes will be drawn first, then the predicted bounding
    boxes.

    Args:
        image: Numpy array of image data. May be of shape (C, H, W) or (H, W, C).
            If array type is not uint8, all values will be clipped between 0.0
            and 1.0 then scaled to a uint8 between 0 and 255.
        ground_truth_boxes: Optional array of shape (N, 4) containing ground truth
            bounding boxes in (xmin, ymin, xmax, ymax) format.
        ground_truth_color: Color to use for ground truth bounding boxes. Color can
            be represented as PIL strings (e.g., "red").
        ground_truth_width: Width of ground truth bounding boxes.
        pred_boxes: Optional array of shape (N, 4) containing predicted
            bounding boxes in (xmin, ymin, xmax, ymax) format.
        pred_color: Color to use for predicted bounding boxes. Color can
            be represented as PIL strings (e.g., "red").
        pred_width: Width of ground truth bounding boxes.

    Return:
        Numpy uint8 array of (C, H, W) image with bounding boxes data
    """
    if image.shape[-1] in (1, 3, 6):  # Convert from (H, W, C) to (C, H, W)
        image = image.transpose(2, 0, 1)

    if image.dtype != np.uint8:  # Convert/scale to uint8
        if np.max(image) <= 1:
            image = np.round(np.clip(image, 0.0, 1.0) * 255.0)
        image = image.astype(np.uint8)

    with_boxes = torch.as_tensor(image)

    if ground_truth_boxes is not None and len(ground_truth_boxes) > 0:
        with_boxes = draw_bounding_boxes(
            image=with_boxes,
            boxes=torch.as_tensor(ground_truth_boxes),
            colors=ground_truth_color,
            width=ground_truth_width,
        )

    if pred_boxes is not None and len(pred_boxes) > 0:
        with_boxes = draw_bounding_boxes(
            image=with_boxes,
            boxes=torch.as_tensor(pred_boxes),
            colors=pred_color,
            width=pred_width,
        )

    return with_boxes.numpy()
