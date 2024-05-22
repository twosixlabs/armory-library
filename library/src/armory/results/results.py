"""Armory evaluation results"""

from collections import UserDict
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    import mlflow.client
    import mlflow.entities
    import rich.console


class EvaluationResults:
    """Armory evaluation results corresponding to a single MLFlow run"""

    def __init__(
        self, client: "mlflow.client.MlflowClient", run: "mlflow.entities.Run"
    ):
        self._client = client
        self._run = run

    @property
    def run_id(self) -> str:
        """MLFlow run ID"""
        return self._run.info.run_id

    @property
    def run_name(self) -> str:
        """MLFlow run name"""
        return self._run.info.run_name or ""

    @cached_property
    def details(self) -> "RunDataDict":
        """Run details"""
        info = self._run.info
        return RunDataDict(
            data={
                k: getattr(info, k, "")
                for k in info.__dir__()
                if k[0] != "_" and type(getattr(info, k, "")).__name__ != "method"
            },
            label="Details",
        )

    @cached_property
    def params(self) -> "RunDataDict":
        """Run parameters"""
        return RunDataDict(data=self._run.data.params, label="Parameters")

    @cached_property
    def tags(self) -> "RunDataDict":
        """Run tags"""
        return RunDataDict(data=self._run.data.tags, label="Tags")

    @cached_property
    def metrics(self) -> "RunDataDict":
        """Run metrics"""
        return RunDataDict(data=self._run.data.metrics, label="Metrics")

    @cached_property
    def children(self) -> Dict[str, "EvaluationResults"]:
        """Child runs"""
        runs = self._client.search_runs(
            experiment_ids=[self._run.info.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{self.run_id}'",
        )
        return {
            run.info.run_name or run.info.run_id: EvaluationResults(self._client, run)
            for run in runs
        }


class RunDataDict(UserDict[str, Any]):
    """Dictionary of run data that can be printed as a table"""

    def __init__(self, data: Dict[str, Any], label: str):
        """
        Initializes the data dictionary.

        Args:
            data: Dictionary contents
            label: Label for the data
        """
        super().__init__(data)
        self.label = label

    def table(
        self,
        console: Optional["rich.console.Console"] = None,
        key_label: str = "key",
        title: Optional[str] = None,
        value_label: str = "value",
        **kwargs,
    ) -> None:
        """
        Prints the contents of the dictionary in a rich table.

        Args:
            console: Optional, rich console to use for printing. Defaults to the
                standard rich console.
            key_label: Optional, label for the key column
            title: Optional, title for the table. Defaults to the dictionary
                label.
            value_label: Optional, label for the value column
            **kwargs: Keyword arguments forwarded to the rich table constructor
        """
        from rich.table import Table

        table = Table(title=title or self.label, **kwargs)
        table.add_column(key_label, style="cyan", no_wrap=True)
        table.add_column(value_label, style="magenta")

        for key, value in sorted(self.items()):
            table.add_row(key, str(value))

        if console is None:
            from rich.console import Console

            console = Console()

        console.print(table)
