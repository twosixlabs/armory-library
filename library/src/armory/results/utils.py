from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlflow.client


def get_mlflow_client() -> "mlflow.client.MlflowClient":
    """Create an MLFlow client"""
    from mlflow.client import MlflowClient

    from armory.evaluation import SysConfig
    from armory.track import init_tracking_uri

    sysconfig = SysConfig()
    tracking_uri = init_tracking_uri(sysconfig.armory_home)
    return MlflowClient(tracking_uri=tracking_uri)
