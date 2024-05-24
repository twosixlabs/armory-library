"""Armory evaluation results"""

from collections import UserDict
from functools import cached_property
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    import mlflow.client
    import mlflow.entities
    import rich.console


_NEXT_PORT = int(os.getenv("PORT", "8050"))


def _get_next_port() -> str:
    global _NEXT_PORT
    port = _NEXT_PORT
    _NEXT_PORT += 1
    return str(port)


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
            title="Details",
        )

    @cached_property
    def params(self) -> "RunDataDict":
        """Run parameters"""
        return RunDataDict(data=self._run.data.params, title="Parameters")

    @cached_property
    def tags(self) -> "RunDataDict":
        """Run tags"""
        return RunDataDict(data=self._run.data.tags, title="Tags")

    @cached_property
    def metrics(self) -> "RunDataDict":
        """Run metrics"""
        return RunDataDict(data=self._run.data.metrics, title="Metrics")

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

    def __init__(
        self,
        data: Dict[str, Any],
        title: str,
        key_label: str = "key",
        value_label: str = "value",
    ):
        """
        Initializes the data dictionary.

        Args:
            data: Dictionary contents
            title: Title for the dictionary when printed as a table
            key_label: Optional, label for the key column when printed as a table
            value_label: Optional, label for the value column when printed as a table
        """
        super().__init__(data)
        self.title = title
        self.key_label = key_label
        self.value_label = value_label

    def table(
        self,
        console: Optional["rich.console.Console"] = None,
        **kwargs,
    ) -> None:
        """
        Prints the contents of the dictionary in a rich table.

        Args:
            console: Optional, rich console to use for printing. Defaults to the
                standard rich console.
            **kwargs: Keyword arguments forwarded to the rich table constructor
        """
        from rich.table import Table

        table = Table(title=self.title, **kwargs)
        table.add_column(self.key_label, style="cyan", no_wrap=True)
        table.add_column(self.value_label, style="magenta")

        for key, value in sorted(self.items()):
            table.add_row(key, str(value))

        if console is None:
            from rich.console import Console

            console = Console()

        console.print(table)

    def plot(self, debug: bool = False, height: int = 400, port: Optional[int] = None):
        import dash

        data = [{"key": key, "value": value} for key, value in sorted(self.items())]

        app = dash.Dash()
        app.layout = dash.html.Div(
            children=[
                dash.html.H1(children=self.title, style={"textAlign": "center"}),
                dash.dash_table.DataTable(
                    data=data,
                    columns=[
                        {"id": "key", "name": self.key_label},
                        {"id": "value", "name": self.value_label},
                    ],
                    style_header={
                        "fontWeight": "bold",
                        "backgroundColor": "rgb(230, 230, 230)",
                        "textAlign": "center",
                    },
                    style_cell={
                        "overflow": "hidden",
                        "textAlign": "left",
                        "textOverflow": "ellipsis",
                    },
                    style_cell_conditional=[
                        {"if": {"column_id": "key"}, "width": "20%"},
                        {"if": {"column_id": "value"}, "maxWidth": 0},
                    ],
                    style_data_conditional=[
                        {
                            "if": {"row_index": "odd"},
                            "backgroundColor": "rgb(240, 240, 240)",
                        },
                    ],
                    tooltip_data=[{"value": str(e["value"])} for e in data],
                    tooltip_delay=1000,
                    tooltip_duration=None,
                ),
            ],
            style={"background": "white"},
        )

        app.run(
            debug=debug,
            jupyter_height=height,
            port=str(port) if port is not None else _get_next_port(),
        )

    def _ipython_display_(self):
        self.plot()
