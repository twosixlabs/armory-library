import json
import os
from pathlib import Path
import tempfile
from typing import List

import matplotlib.pyplot as plt
from mlflow.client import MlflowClient
import numpy as np

from armory.evaluation import SysConfig
from armory.track import init_tracking_uri


def suppress_artifact_progress_bar():
    os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "False"


def get_mlflow_client():
    sysconfig = SysConfig()
    tracking_uri = init_tracking_uri(sysconfig.armory_home)
    return MlflowClient(tracking_uri=tracking_uri)


def get_predicted_label(filepath: Path, labels: List[str]):
    with open(filepath, "r") as infile:
        data = json.load(infile)
        if "y_predicted" not in data:
            return "unknown"
        y_predicted = data["y_predicted"]
        index = np.argmax(y_predicted)
        return f"{labels[index]} ({index})"


def display_image_classification_results(
    run_id: str, batch_idx: int, batch_size: int, chains: List[str], labels: List[str]
):
    suppress_artifact_progress_bar()
    client = get_mlflow_client()

    fig, axes = plt.subplots(
        nrows=batch_size,
        ncols=len(chains),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        for sample_idx in range(batch_size):
            for chain_idx, chain in enumerate(chains):
                image_filename = f"batch_{batch_idx}_ex_{sample_idx}_{chain}.png"
                client.download_artifacts(run_id, image_filename, tmpdir)
                image = plt.imread(tmppath / image_filename)

                json_filename = f"batch_{batch_idx}_ex_{sample_idx}_{chain}.txt"
                client.download_artifacts(run_id, json_filename, tmpdir)
                predicted_label = get_predicted_label(tmppath / json_filename, labels)

                ax = axes[sample_idx][chain_idx]
                if sample_idx == 0:
                    ax.set_title(chain)
                if chain_idx == 0:
                    ax.set_ylabel(f"Sample {sample_idx}")
                ax.imshow(image)
                ax.set_xlabel(f"Predicted: {predicted_label}")
                ax.tick_params(
                    bottom=False, left=False, labelbottom=False, labelleft=False
                )

    fig.suptitle(f"Batch {batch_idx}")
    fig.tight_layout()
