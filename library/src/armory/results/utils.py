import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlflow.client


def get_mlflow_client() -> "mlflow.client.MlflowClient":
    from mlflow.client import MlflowClient

    from armory.evaluation import SysConfig
    from armory.track import init_tracking_uri

    sysconfig = SysConfig()
    tracking_uri = init_tracking_uri(sysconfig.armory_home)
    return MlflowClient(tracking_uri=tracking_uri)


_NEXT_DASH_PORT = int(os.getenv("PORT", "8050"))


def get_next_dash_port() -> str:
    global _NEXT_DASH_PORT
    port = _NEXT_DASH_PORT
    _NEXT_DASH_PORT += 1
    return str(port)
