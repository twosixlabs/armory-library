import os

from mlflow.client import MlflowClient

from armory.evaluation import SysConfig
from armory.track import init_tracking_uri


def get_mlflow_client():
    sysconfig = SysConfig()
    tracking_uri = init_tracking_uri(sysconfig.armory_home)
    return MlflowClient(tracking_uri=tracking_uri)


_NEXT_DASH_PORT = int(os.getenv("PORT", "8050"))


def get_next_dash_port() -> str:
    global _NEXT_DASH_PORT
    port = _NEXT_DASH_PORT
    _NEXT_DASH_PORT += 1
    return str(port)


# def for_experiment(
#     name: Optional[str] = None,
#     experiment_id: Optional[str] = None,
#     max_results: int = 100,
# ) -> Sequence[EvaluationResults]:
#     client = _get_mlflow_client()
#     if name is not None and experiment_id is None:
#         experiments = client.search_experiments(filter_string=f"name = '{name}'")
#         if len(experiments) == 1:
#             experiment_id = experiments[0].experiment_id
#         else:
#             raise RuntimeError(f"No experiment found with name '{name}'")
#     if experiment_id is None:
#         raise ValueError("No experiment ID or name provided")

#     runs = client.search_runs(experiment_ids=[experiment_id], max_results=max_results)
#     if len(runs) == 0:
#         raise RuntimeError("No runs found for experiment")
#     return [EvaluationResults(client, run) for run in runs]
