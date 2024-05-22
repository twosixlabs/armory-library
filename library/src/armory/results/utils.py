from mlflow.client import MlflowClient

from armory.evaluation import SysConfig
from armory.results.results import EvaluationResults
from armory.track import init_tracking_uri


def _get_mlflow_client():
    sysconfig = SysConfig()
    tracking_uri = init_tracking_uri(sysconfig.armory_home)
    return MlflowClient(tracking_uri=tracking_uri)


def for_run(run_id: str) -> EvaluationResults:
    client = _get_mlflow_client()
    return EvaluationResults(client, client.get_run(run_id))
