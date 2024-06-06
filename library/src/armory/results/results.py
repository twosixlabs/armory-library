"""Armory evaluation results"""

from collections import UserDict
from functools import cached_property
import json
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Protocol,
    Sequence,
    Union,
)

from armory.results.utils import get_mlflow_client, get_next_dash_port

if TYPE_CHECKING:
    import PIL.Image
    import matplotlib.figure
    import matplotlib.pyplot
    import mlflow.client
    import mlflow.entities
    import rich.console


class Plottable(Protocol):
    def plot(self, *args, **kwargs) -> "matplotlib.figure.Figure": ...


class EvaluationResults:
    """Armory evaluation results corresponding to a single MLFlow run"""

    @classmethod
    def for_run(cls, run_id: str) -> "EvaluationResults":
        client = get_mlflow_client()
        return cls(client, client.get_run(run_id))

    def __init__(
        self, client: "mlflow.client.MlflowClient", run: "mlflow.entities.Run"
    ):
        self._client = client
        self._run = run

    def __repr__(self) -> str:
        return f"EvaluationResults(client={self._client}, run={self._run})"

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
    def metrics(self) -> "RunMetricsDict":
        """Run model metrics"""
        return RunMetricsDict(
            data={
                k: v
                for k, v in self._run.data.metrics.items()
                if not k.startswith("system/")
            },
            title="Metrics",
        )

    @cached_property
    def system_metrics(self) -> "RunMetricsDict":
        """Run system metrics"""
        return RunMetricsDict(
            data={
                k: v
                for k, v in self._run.data.metrics.items()
                if k.startswith("system/")
            },
            title="System Metrics",
        )

    @cached_property
    def artifacts(self) -> "RunArtifacts":
        return RunArtifacts(self._client, self.run_id)

    @cached_property
    def children(self) -> Dict[str, "EvaluationResults"]:
        """Child runs"""
        runs = self._client.search_runs(
            experiment_ids=[self._run.info.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{self.run_id}'",
        )
        runs = sorted(list(runs), key=lambda run: str(run.info.run_name))
        return {
            run.info.run_name or run.info.run_id: EvaluationResults(self._client, run)
            for run in runs
        }

    @cached_property
    def batches(self) -> Iterable[int]:
        return set(
            sorted([int(p.split("/")[1]) for p in self.artifacts["exports/"].paths()])
        )

    def batch(self, batch_idx: int) -> "BatchExports":
        return BatchExports(batch_idx, self.artifacts[f"exports/{batch_idx:05}"])


class RunDataDict(UserDict):
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
        format: Callable[[Any], str] = str,
        title: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Prints the contents of the dictionary in a rich table.

        Args:
            console: Optional, rich console to use for printing. Defaults to the
                standard rich console.
            format: Optional, function to format values for printing. Defaults to
                the built-in str function.
            **kwargs: Keyword arguments forwarded to the rich table constructor
        """
        from rich.table import Table

        table = Table(title=title or self.title, **kwargs)
        table.add_column(self.key_label, style="cyan", no_wrap=True)
        table.add_column(self.value_label, style="magenta")

        for key, value in sorted(self.items()):
            table.add_row(key, format(value))

        if console is None:
            from rich.console import Console

            console = Console()

        console.print(table)

    def plot(
        self,
        align_left: bool = True,
        dark: bool = False,
        debug: bool = False,
        format: Callable[[Any], str] = str,
        height: int = 400,
        port: Optional[int] = None,
    ) -> None:
        """
        Displays the contents of the dictionary in an HTML table.

        Args:
            align_left: Optional, align value column text to the left if True
            dark: Optional, use a dark theme if True
            debug: Optional, enable debug output from Dash
            format: Optional, function to format values for printing. Defaults to
                the built-in str function.
            height: Optional, height of the table in pixels
            port: Optional, port to use for the Dash server. Defaults to the next
                available port.
        """
        import dash

        data = [
            {"key": key, "value": format(value)} for key, value in sorted(self.items())
        ]

        app = dash.Dash()
        app.layout = dash.html.Div(
            children=[
                dash.html.H1(
                    children=self.title,
                    style={
                        "color": "white" if dark else "black",
                        "textAlign": "center",
                    },
                ),
                dash.dash_table.DataTable(
                    data=data,
                    columns=[
                        {"id": "key", "name": self.key_label},
                        {"id": "value", "name": self.value_label},
                    ],
                    cell_selectable=False,
                    style_header={
                        "fontWeight": "bold",
                        "backgroundColor": (
                            "rgb(10, 10, 10)" if dark else "rgb(229, 229, 229)"
                        ),
                        "color": "white" if dark else "black",
                        "textAlign": "center",
                    },
                    style_cell={
                        "backgroundColor": "rgb(30, 30, 30)" if dark else "white",
                        "border": (
                            "1px solid dimgray" if dark else "1px solid lightgray"
                        ),
                        "color": "white" if dark else "black",
                        "overflow": "hidden",
                        "textAlign": "left",
                        "textOverflow": "ellipsis",
                    },
                    style_cell_conditional=[
                        {"if": {"column_id": "key"}, "width": "20%"},
                        {
                            "if": {"column_id": "value"},
                            "maxWidth": 0,
                            "textAlign": "left" if align_left else "right",
                        },
                    ],
                    style_data_conditional=[
                        {
                            "if": {"row_index": "odd"},
                            "backgroundColor": (
                                "rgb(23, 23, 23)" if dark else "rgb(250, 250, 250)"
                            ),
                        },
                    ],
                    tooltip_data=[{"value": str(e["value"])} for e in data],
                    tooltip_delay=1000,
                    tooltip_duration=None,
                ),
            ],
            style={"background": "rgb(30, 30, 30)" if dark else "white"},
        )

        app.run(
            debug=debug,
            jupyter_height=height,
            port=str(port) if port is not None else get_next_dash_port(),
        )

    def _ipython_display_(self):
        self.plot()


class RunMetricsDict(RunDataDict):

    def __init__(
        self,
        data: Dict[str, Any],
        title: str,
        key_label: str = "metric",
        value_label: str = "value",
    ):
        """
        Initializes the metrics dictionary.

        Args:
            data: Dictionary contents
            title: Title for the dictionary when printed as a table
            key_label: Optional, label for the key column when printed as a table
            value_label: Optional, label for the value column when printed as a table
        """
        super().__init__(
            data=data, title=title, key_label=key_label, value_label=value_label
        )

    def table(
        self,
        console: Optional["rich.console.Console"] = None,
        format: Optional[Callable[[Any], str]] = None,
        precision: int = 3,
        **kwargs,
    ) -> None:
        """
        Prints the contents of the dictionary in a rich table.

        Args:
            console: Optional, rich console to use for printing. Defaults to the
                standard rich console.
            format: Optional, function to format values for printing. Defaults to
                a fixed-precision floating point formatter.
            precision: Optional, number of decimal places to display for floating
                point values
            **kwargs: Keyword arguments forwarded to the rich table constructor
        """
        return super().table(
            console=console,
            format=format or (lambda v: f"{v:.{precision}f}"),
            **kwargs,
        )

    def plot(
        self,
        align_left: bool = False,
        dark: bool = False,
        debug: bool = False,
        format: Optional[Callable[[Any], str]] = None,
        height: int = 400,
        precision: int = 3,
        port: Optional[int] = None,
    ) -> None:
        """
        Displays the contents of the dictionary in an HTML table.

        Args:
            align_left: Optional, align value column text to the left if True
            dark: Optional, use a dark theme if True
            debug: Optional, enable debug output from Dash
            format: Optional, function to format values for printing. Defaults to
                a fixed-precision floating point formatter.
            height: Optional, height of the table in pixels
            precision: Optional, number of decimal places to display for floating
                point values
            port: Optional, port to use for the Dash server. Defaults to the next
                available port.
        """
        return super().plot(
            align_left=align_left,
            dark=dark,
            debug=debug,
            format=format or (lambda v: f"{v:.{precision}f}"),
            height=height,
            port=port,
        )


class RunArtifacts:

    def __init__(
        self,
        client: "mlflow.client.MlflowClient",
        run_id: str,
        path: str = "",
        children: Optional[Dict[str, "RunArtifact"]] = None,
    ):
        self._client = client
        self._run_id = run_id
        self.path = path
        self._children = (
            children if children is not None else self._list_artifacts(path)
        )

    def __repr__(self) -> str:
        return f"RunArtifacts(client={self._client}, run_id={self._run_id}, path={self.path}, children={self._children})"

    def paths(self) -> Iterable[str]:
        return self._children.keys()

    def _list_artifacts(self, parent_path) -> Dict[str, "RunArtifact"]:
        artifacts: Dict[str, RunArtifact] = {}
        for child in self._client.list_artifacts(self._run_id, parent_path):
            if child.is_dir:
                artifacts.update(self._list_artifacts(child.path))
            else:
                artifacts[child.path] = RunArtifact(self._client, self._run_id, child)
        return artifacts

    def __getitem__(self, key: str) -> Union["RunArtifact", "RunArtifacts"]:
        if key[-1] == "/":  # remove trailing /, if any
            key = key[:-1]
        path = self.path + "/" + key if self.path else key
        item = self._children.get(path, None)
        if item:  # if path matches a child artifact, return it
            return item
        # else return a new artifacts parent with all items under the path
        prefix = path + "/"
        children = {k: v for k, v in self._children.items() if k.startswith(prefix)}
        return RunArtifacts(self._client, self._run_id, path, children)


class RunArtifact:

    def __init__(
        self,
        client: "mlflow.client.MlflowClient",
        run_id: str,
        artifact: "mlflow.entities.FileInfo",
    ):
        self._client = client
        self._run_id = run_id
        self.artifact = artifact

    def __repr__(self) -> str:
        return f"RunArtifact(client={self._client}, run_id={self._run_id}, artifact={self.artifact})"

    @cached_property
    def local_path(self) -> str:
        return self._client.download_artifacts(self._run_id, self.artifact.path)

    @cached_property
    def data(self) -> bytes:
        with open(self.local_path, "rb") as f:
            return f.read()

    @cached_property
    def image(self) -> "PIL.Image.Image":
        from PIL import Image

        return Image.open(self.local_path)

    @cached_property
    def json(self) -> Any:
        with open(self.local_path, "r") as f:
            return json.load(f)


class BatchExports:

    def __init__(
        self,
        batch_idx: int,
        artifacts: RunArtifacts,
    ):
        self.batch_idx = batch_idx
        self.artifacts = artifacts

    def __repr__(self) -> str:
        return f"BatchExports(batch_idx={self.batch_idx}, artifacts={self.artifacts})"

    @cached_property
    def samples(self) -> Iterable[int]:
        return set(sorted([int(p.split("/")[2]) for p in self.artifacts.paths()]))

    def sample(self, sample_idx: int) -> "SampleExports":
        return SampleExports(
            self.batch_idx,
            sample_idx,
            self.artifacts[f"{sample_idx:02}"],
        )

    def plot(
        self,
        filename: Optional[str] = None,
        max_samples: Optional[int] = None,
        samples: Optional[Sequence[int]] = None,
        title: Optional[str] = None,
    ) -> "matplotlib.figure.Figure":
        from armory.results.plots import plot_in_grid

        sample_nums = []
        for idx, sample_num in enumerate(self.samples):
            if max_samples and idx == max_samples:
                break
            elif samples and sample_num not in samples:
                continue
            sample_nums.append(sample_num)

        if filename is None:
            filename = self.sample(sample_nums[0]).imagename

        return plot_in_grid(
            [self.sample(sample_num)[filename].image for sample_num in sample_nums],
            rows=[f"Sample {sample_num}" for sample_num in sample_nums],
            title=title or f"Batch {self.batch_idx}",
            vertical=True,
        )


class SampleExports:

    def __init__(
        self,
        batch_idx: int,
        sample_idx: int,
        artifacts: RunArtifacts,
    ):
        self.batch_idx = batch_idx
        self.sample_idx = sample_idx
        self.artifacts = artifacts

    def __repr__(self) -> str:
        return f"SampleExports(batch_idx={self.batch_idx}, sample_idx={self.sample_idx}, artifacts={self.artifacts})"

    @property
    def classification(self) -> "ClassificationResults":
        return ClassificationResults(self)

    @cached_property
    def exports(self) -> Iterable[str]:
        return sorted([p.split("/")[-1] for p in self.artifacts.paths()])

    @cached_property
    def imagename(self) -> str:
        for path in self.exports:
            if path in ("input.png", "objects.png"):
                return path
        raise ValueError(
            f"No default image export found for sample {self.sample_idx} in batch {self.batch_idx}"
        )

    @cached_property
    def metadata(self) -> "SampleMetadata":
        return SampleMetadata(
            self.batch_idx, self.sample_idx, self.artifacts["metadata.txt"]
        )

    def __getitem__(self, key: str) -> RunArtifact:
        return self.artifacts[key]


class SampleMetadata:

    def __init__(
        self,
        batch_idx: int,
        sample_idx: int,
        artifact: RunArtifact,
    ):
        self.batch_idx = batch_idx
        self.sample_idx = sample_idx
        self.artifact = artifact

    @property
    def json(self) -> Any:
        return self.artifact.json

    def __getitem__(self, key: str) -> Any:
        return self.json.get(key)


class ClassificationResults:

    def __init__(
        self,
        sample: SampleExports,
    ):
        self.sample = sample

    def __repr__(self) -> str:
        return f"ClassificationResults(sample={self.sample})"

    def plot(
        self,
        figure: Optional["matplotlib.pyplot.Figure"] = None,
        labels: Optional[Sequence[str]] = None,
        top_k: int = 10,
    ) -> "matplotlib.figure.Figure":
        import matplotlib.pyplot as plt
        import numpy as np

        with plt.ioff():
            if figure is None:
                figure = plt.figure()

            (ax1, ax2) = figure.subplots(1, 2)

            ax1.imshow(self.sample[self.sample.imagename].image)
            ax1.axis("off")

            target_class = self.sample.metadata["targets"]

            probs = np.array(self.sample.metadata["predictions"])
            if np.max(probs) > 1 or np.min(probs) < 0:
                # perform softmax to turn logits into probabilities
                probs = np.exp(probs) / np.sum(np.exp(probs))
            top_ten_indices = list(probs.argsort()[-top_k:][::-1])
            top_probs = probs[top_ten_indices]

            barlist = ax2.bar(range(top_k), top_probs)
            if target_class in top_ten_indices:
                barlist[top_ten_indices.index(target_class)].set_color("g")

            if labels is not None:
                barlabels = [labels[i] for i in top_ten_indices]
            else:
                barlabels = [str(i) for i in top_ten_indices]

            ax2.set_ylim([0, 1.1])
            ax2.set_xticks(range(top_k))
            ax2.set_xticklabels(barlabels, rotation="vertical")
            ax2.set_ylabel("Probability")

            figure.subplots_adjust(bottom=0.2, wspace=0.3)
            return figure
