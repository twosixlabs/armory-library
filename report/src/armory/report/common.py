import json
import os
import pathlib
from typing import List

import mlflow.client


def create_client():
    tracking_uri = None
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        tracking_uri = str(pathlib.Path.home() / ".armory" / "mlruns")
    return mlflow.client.MlflowClient(tracking_uri=tracking_uri)


def _serialize(obj):
    try:
        return json.loads(json.dumps(obj))
    except TypeError:
        return {
            k: _serialize(getattr(obj, k, ""))
            for k in obj.__dir__()
            if k[0] != "_" and type(getattr(obj, k, "")).__name__ != "method"
        }


def dump_experiment(experiment_id: str):
    client = create_client()
    experiment = client.get_experiment(experiment_id)
    return {
        "experiment": _serialize(experiment),
        "runs": [
            _serialize(run)
            for run in client.search_runs(experiment_ids=[experiment_id])
        ],
    }


def dump_runs(run_ids: List[str]):
    client = create_client()
    data = {"runs": [_serialize(client.get_run(run_id)) for run_id in run_ids]}
    experiment_ids = set([run["info"]["experiment_id"] for run in data["runs"]])
    if len(experiment_ids) == 1:
        data["experiment"] = _serialize(client.get_experiment(experiment_ids.pop()))
    else:
        data["experiment"] = None
    return data
