import json
import os
import pathlib
import tempfile
from typing import List, Optional

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


def dump_artifacts(
    run_id: str,
    batches: List[str],
    max_samples: Optional[int],
    extension: str,
    outdir: pathlib.Path,
):
    artifacts = dict()
    outdir.mkdir(parents=True, exist_ok=True)
    client = create_client()

    with tempfile.TemporaryDirectory() as tmpdir:
        for artifact in client.list_artifacts(run_id):
            segments = artifact.path.split("_", 4)
            if len(segments) != 5:
                continue
            (_, batch, _, sample, remainder) = segments
            if batch not in batches:
                continue
            if max_samples and int(sample) >= max_samples:
                continue

            segments = remainder.split(".", 1)
            if len(segments) != 2:
                continue
            (chain, ext) = segments
            if ext != extension:
                continue

            artifacts.setdefault(chain, dict())
            artifacts[chain].setdefault(batch, dict())
            artifacts[chain][batch][sample] = dict(file=artifact.path)

            client.download_artifacts(run_id, artifact.path, str(outdir))

            metadatapath = artifact.path.replace(extension, "txt")
            client.download_artifacts(run_id, metadatapath, tmpdir)

            with open(os.path.join(tmpdir, metadatapath), "r") as jsonfile:
                artifacts[chain][batch][sample].update(json.load(jsonfile))

    return artifacts
