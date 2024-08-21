"""Armory evaluation results"""

from collections import UserDict, UserList
from functools import cached_property
import json
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Sequence,
    Union,
    cast,
)

from armory.results.utils import get_mlflow_client

if TYPE_CHECKING:
    import IPython.core.display
    import PIL.Image
    import matplotlib.figure
    import matplotlib.pyplot
    import mlflow.client
    import mlflow.entities
    import rich.console


class EvaluationResults:
    """Armory evaluation results corresponding to a single MLFlow run"""

    @classmethod
    def for_run(cls, run_id: str) -> "EvaluationResults":
        """
        Retrieve the evaluation results for a given MLFlow run ID.

        Args:
            run_id: MLFlow run ID

        Return:
            EvaluationResults object
        """
        client = get_mlflow_client()
        return cls(client, client.get_run(run_id))

    @classmethod
    def for_last_run(
        cls,
        experiment_id: Optional[str] = None,
        experiment_name: Optional[str] = None,
        max_search: int = 10,
    ) -> "EvaluationResults":
        """
        Retrieve the evaluation results for the last run in a given MLFlow
        experiment (by name or by ID).

        Args:
            experiment_id: Optional, MLFlow experiment ID (if not using name)
            experiment_name: Optional, MLFlow experiment name (if not using ID)
            max_search: Optional, number of runs to search. This should only be
                necessary if the evaluations have more than 10 chains.

        Return:
            EvaluationResults object
        """
        client = get_mlflow_client()
        if experiment_id:
            experiment = client.get_experiment(experiment_id)
        elif experiment_name:
            experiment = client.get_experiment_by_name(experiment_name)
        else:
            raise ValueError("Either experiment_id or experiment_name must be provided")
        if experiment is None:
            raise ValueError(f"Experiment not found: {experiment_name}")
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=max_search,
            order_by=["start_time DESC"],
        )
        # Have to filter out child runs here instead of via filter_string,
        # see https://github.com/mlflow/mlflow/issues/2922
        runs = [
            run for run in runs if run.data.tags.get("mlflow.parentRunId", None) is None
        ]
        if len(runs) == 0:
            raise ValueError(f"No runs found in experiment: {experiment_name}")
        return cls(client, runs[0])

    def __init__(
        self, client: "mlflow.client.MlflowClient", run: "mlflow.entities.Run"
    ):
        """
        Initialize the evaluation results object for the given MLFlow run.

        Args:
            client: MLFlow client
            run: MLFlow run
        """
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
            client=self._client,
            run_id=self.run_id,
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
            client=self._client,
            run_id=self.run_id,
        )

    @cached_property
    def artifacts(self) -> "RunArtifacts":
        """Run artifacts"""
        return RunArtifacts(self._client, self.run_id)

    @cached_property
    def children(self) -> Dict[str, "EvaluationResults"]:
        """Child (nested) runs"""
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
        """Indices of exported batches in the run"""
        return set(
            sorted(
                [
                    int(p.split("/")[1])
                    for p in cast(RunArtifacts, self.artifacts["exports/"]).paths()
                ]
            )
        )

    def batch(self, batch_idx: int) -> "BatchExports":
        """Retrieve exports for a specific batch index"""
        return BatchExports(
            batch_idx, cast(RunArtifacts, self.artifacts[f"exports/{batch_idx:05}"])
        )


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
        Initialize the data dictionary.

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
        Print the contents of the dictionary in a rich table.

        Args:
            console: Optional, rich console to use for printing. Defaults to the
                standard rich console.
            format: Optional, function to format values for printing. Defaults to
                the built-in str function.
            title: Optional, title for the table. Defaults to the title provided
                when the dictionary was initialized.
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
        format: Callable[[Any], str] = str,
    ) -> "IPython.core.display.HTML":
        """
        Create an HTML table for the dictionary.

        Args:
            format: Optional, function to format values for printing. Defaults to
                the built-in str function.

        Return:
            IPython HTML object for the table
        """
        from IPython.core.display import HTML

        table = "<table>"

        table += "<tr>"
        table += (
            f"<th style='font-weight: bold; text-align: center;'>{self.key_label}</th>"
        )
        table += f"<th style='font-weight: bold; text-align: center;'>{self.value_label}</th>"
        table += "</tr>"

        for key, value in sorted(self.items()):
            table += "<tr>"
            table += f"<td style='font-weight: bold; text-align: left; width: 20%'>{key}</td>"
            table += f"<td style='max-width: 0; overflow: hidden; text-align: left; text-overflow: ellipsis' title='{value}'>"
            table += f"{format(value)}</td>"
            table += "</tr>"

        table += "</table>"

        return HTML(table)

    def _repr_html_(self):
        return self.plot().data


class RunMetricsDict(RunDataDict):
    """Dictionary of run metrics that can be printed as a table"""

    def __init__(
        self,
        data: Dict[str, Any],
        title: str,
        client: "mlflow.client.MlflowClient",
        run_id: str,
        key_label: str = "metric",
        value_label: str = "value",
    ):
        """
        Initialize the metrics dictionary.

        Args:
            data: Dictionary contents
            title: Title for the dictionary when printed as a table
            key_label: Optional, label for the key column when printed as a table
            value_label: Optional, label for the value column when printed as a table
        """
        super().__init__(
            data=data, title=title, key_label=key_label, value_label=value_label
        )
        self._client = client
        self._run_id = run_id

    def history(self, key: str) -> "MetricHistory":
        """
        Retrieve the history of a specific metric.

        Args:
            key: Metric key

        Return:
            Metric history object
        """
        if key not in self.keys():
            raise KeyError(f"Metric not found: {key}")
        return MetricHistory(self._client.get_metric_history(self._run_id, key))

    def table(
        self,
        console: Optional["rich.console.Console"] = None,
        format: Optional[Callable[[Any], str]] = None,
        precision: int = 3,
        title: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Print the contents of the dictionary in a rich table.

        Args:
            console: Optional, rich console to use for printing. Defaults to the
                standard rich console.
            format: Optional, function to format values for printing. Defaults to
                a fixed-precision floating point formatter.
            precision: Optional, number of decimal places to display for floating
                point values
            title: Optional, title for the table. Defaults to the title provided
                when the dictionary was initialized.
            **kwargs: Keyword arguments forwarded to the rich table constructor
        """
        return super().table(
            console=console,
            format=format or (lambda v: f"{v:.{precision}f}"),
            title=title,
            **kwargs,
        )

    def plot(
        self,
        format: Optional[Callable[[Any], str]] = None,
        precision: int = 3,
    ) -> "IPython.core.display.HTML":
        """
        Create an HTML table for the metrics.

        Args:
            format: Optional, function to format values for printing. Defaults to
                a fixed-precision floating point formatter.
            precision: Optional, number of decimal places to display for floating
                point values

        Return:
            IPython HTML object for the table
        """
        return super().plot(
            format=format or (lambda v: f"{v:.{precision}f}"),
        )


class MetricHistory(UserList):

    def __init__(self, history: Sequence["mlflow.entities.Metric"]):
        """
        Initialize the metric history object.

        Args:
            history: List of metric history entries
        """
        super().__init__(history)

    def plot(
        self,
        figure: Optional["matplotlib.figure.Figure"] = None,
        timestamp: bool = False,
    ) -> "matplotlib.figure.Figure":
        """
        Create a matplotlib figure for the metric history.

        Args:
            figure: Optional, existing matplotlib figure to use for plotting.
                If not provided, a new figure will be created.
            timestamp: Use timestamps for the x-axis instead of step numbers

        Return:
            Matplotlib figure
        """
        import matplotlib.pyplot as plt

        with plt.ioff():
            if figure is None:
                figure = plt.figure()

            steps = [entry.timestamp if timestamp else entry.step for entry in self]
            values = [entry.value for entry in self]

            ax = figure.add_subplot()

            ax.plot(steps, values)
            ax.set_xlabel("Timestamp" if timestamp else "Step")
            ax.set_ylabel("Value")
            ax.set_title(self[0].key)

            return figure


class RunArtifacts:
    """Attached artifacts for a specific folder path in an MLFlow run"""

    def __init__(
        self,
        client: "mlflow.client.MlflowClient",
        run_id: str,
        path: str = "",
        children: Optional[Dict[str, "RunArtifact"]] = None,
    ):
        """
        Initialize the artifacts.

        Args:
            client: MLFlow client
            run_id: MLFlow run ID
            path: Optional, the folder path within the MLFlow run's artifacts.
                Default is the root path.
            children: Optional, dictionary of paths to child artifacts. If not
                provided, the children will be populated by querying the MLFlow
                server.
        """
        self._client = client
        self._run_id = run_id
        self.path = path
        self._children = (
            children if children is not None else self._list_artifacts(path)
        )

    def __repr__(self) -> str:
        return f"RunArtifacts(client={self._client}, run_id={self._run_id}, path={self.path}, children={self._children})"

    def paths(self) -> Iterable[str]:
        """Paths of all artifacts under the current path"""
        return self._children.keys()

    def _list_artifacts(self, parent_path) -> Dict[str, "RunArtifact"]:
        """List all artifacts in the current run under a given path"""
        artifacts: Dict[str, RunArtifact] = {}
        for child in self._client.list_artifacts(self._run_id, parent_path):
            if child.is_dir:
                artifacts.update(self._list_artifacts(child.path))
            else:
                artifacts[child.path] = RunArtifact(self._client, self._run_id, child)
        return artifacts

    def __getitem__(self, key: str) -> Union["RunArtifact", "RunArtifacts"]:
        """
        Retrieve the folder path or an individual artifact at the given path.

        Args:
            key: Folder path or artifact name, relative to the current path of
                this object's folder path

        Return:
            RunArtifacts object if the key is a folder path, RunArtifact object
            if the key is an artifact name
        """
        if key[-1] == "/":  # remove trailing /, if any
            key = key[:-1]
        path = self.path + "/" + key if self.path else key
        item = self._children.get(path, None)
        if item:  # if path matches a child artifact, return it
            return item
        # else return a new artifacts parent with all items under the path
        prefix = path + "/"
        children = {k: v for k, v in self._children.items() if k.startswith(prefix)}
        if len(children) == 0:
            raise KeyError(f"Artifact(s) not found: {path}")
        return RunArtifacts(self._client, self._run_id, path, children)


class RunArtifact:
    """An individual artifact attached to an MLFlow run"""

    def __init__(
        self,
        client: "mlflow.client.MlflowClient",
        run_id: str,
        artifact: "mlflow.entities.FileInfo",
    ):
        """
        Initialize the artifact.

        Args:
            client: MLFlow client
            run_id: MLFlow run ID
            artifact: MLFlow file info object
        """
        self._client = client
        self._run_id = run_id
        self.artifact = artifact

    def __repr__(self) -> str:
        return f"RunArtifact(client={self._client}, run_id={self._run_id}, artifact={self.artifact})"

    @cached_property
    def local_path(self) -> str:
        """
        Local path to the downloaded artifact file. If the file is not already
        available locally, it will be downloaded from the MLFlow server.
        """
        return self._client.download_artifacts(self._run_id, self.artifact.path)

    @cached_property
    def data(self) -> bytes:
        """Raw data contents of the artifact file"""
        with open(self.local_path, "rb") as f:
            return f.read()

    @cached_property
    def image(self) -> "PIL.Image.Image":
        """Artifact file as a PIL image"""
        from PIL import Image

        return Image.open(self.local_path)

    @cached_property
    def json(self) -> Any:
        """Artifact file as a parsed JSON object"""
        with open(self.local_path, "r") as f:
            return json.load(f)


class BatchExports:
    """Exported artifacts for a specific batch in an evaluation run"""

    def __init__(
        self,
        batch_idx: int,
        artifacts: RunArtifacts,
    ):
        """
        Initialize the batch exports.

        Args:
            batch_idx: Batch index
            artifacts: Exported artifacts for the batch
        """
        self.batch_idx = batch_idx
        self.artifacts = artifacts

    def __repr__(self) -> str:
        return f"BatchExports(batch_idx={self.batch_idx}, artifacts={self.artifacts})"

    @cached_property
    def samples(self) -> Iterable[int]:
        """Indices of exported samples in the batch"""
        return set(sorted([int(p.split("/")[2]) for p in self.artifacts.paths()]))

    def sample(self, sample_idx: int) -> "SampleExports":
        """Retrieve exports for a specific sample index"""
        return SampleExports(
            self.batch_idx,
            sample_idx,
            cast(RunArtifacts, self.artifacts[f"{sample_idx:02}"]),
        )

    def plot(
        self,
        filename: Optional[str] = None,
        max_samples: Optional[int] = None,
        samples: Optional[Sequence[int]] = None,
        title: Optional[str] = None,
    ) -> "matplotlib.figure.Figure":
        """
        Create a matplotlib figure for image samples in the batch.

        Args:
            filename: Optional, image filename to plot. If not provided, the
                default image export for each sample will be plotted.
            max_samples: Optional, maximum number of samples to plot. If not
                provided, all samples will be plotted.
            samples: Optional, specific sample indices to plot. If not provided,
                all samples will be plotted.
            title: Optional, title for the plot

        Return:
            Matplotlib figure
        """
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

    def _ipython_display_(self):
        from IPython.display import display

        display(self.plot())


class SampleExports:
    """Exported artifacts for a specific sample within a batch"""

    def __init__(
        self,
        batch_idx: int,
        sample_idx: int,
        artifacts: RunArtifacts,
    ):
        """
        Initialize the sample exports.

        Args:
            batch_idx: Batch index
            sample_idx: Sample index
            artifacts: Exported artifacts for the sample
        """
        self.batch_idx = batch_idx
        self.sample_idx = sample_idx
        self.artifacts = artifacts

    def __repr__(self) -> str:
        return f"SampleExports(batch_idx={self.batch_idx}, sample_idx={self.sample_idx}, artifacts={self.artifacts})"

    @property
    def classification(self) -> "ClassificationResults":
        """Sample as an image classification sample"""
        return ClassificationResults(self)

    @cached_property
    def exports(self) -> Iterable[str]:
        """Artifact filenames for the sample"""
        return sorted([p.split("/")[-1] for p in self.artifacts.paths()])

    @cached_property
    def imagename(self) -> str:
        """Default image export for the sample"""
        for path in self.exports:
            if path in ("input.png", "objects.png"):
                return path
        raise ValueError(
            f"No default image export found for sample {self.sample_idx} in batch {self.batch_idx}"
        )

    @cached_property
    def metadata(self) -> "SampleMetadata":
        """Metadata properties for the sample"""
        return SampleMetadata(
            self.batch_idx,
            self.sample_idx,
            cast(RunArtifact, self.artifacts["metadata.txt"]),
        )

    def __getitem__(self, key: str) -> RunArtifact:
        """
        Retrieve the individual artifact at the given path.

        Args:
            key: Artifact filename

        Return:
            RunArtifact object
        """
        return cast(RunArtifact, self.artifacts[key])


class SampleMetadata:
    """Metadata properties for a specific sample within a batch"""

    def __init__(
        self,
        batch_idx: int,
        sample_idx: int,
        artifact: RunArtifact,
    ):
        """
        Initialize the sample metadata.

        Args:
            batch_idx: Batch index
            sample_idx: Sample index
            artifact: Metadata artifact for the sample
        """
        self.batch_idx = batch_idx
        self.sample_idx = sample_idx
        self.artifact = artifact

    @property
    def json(self) -> Any:
        """Sample metadata as a parsed JSON object"""
        return self.artifact.json

    def __getitem__(self, key: str) -> Any:
        """
        Retrieve a specific metadata property.

        Args:
            key: Metadata property key

        Return:
            Metadata property value
        """
        return self.json.get(key)


class ClassificationResults:
    """Image classification results for a specific sample within a batch"""

    def __init__(
        self,
        sample: SampleExports,
    ):
        """
        Initialize the classification results.

        Args:
            sample: Sample exports
        """
        self.sample = sample

    def __repr__(self) -> str:
        return f"ClassificationResults(sample={self.sample})"

    def plot(
        self,
        figure: Optional["matplotlib.pyplot.Figure"] = None,
        labels: Optional[Sequence[str]] = None,
        top_k: int = 10,
    ) -> "matplotlib.figure.Figure":
        """
        Create a matplotlib figure for the image classification input and
        predictions for this sample.

        Args:
            figure: Optional, existing matplotlib figure to use for plotting.
                If not provided, a new figure will be created.
            labels: Optional, class labels for the predictions. If not provided,
                class indices will be used.
            top_k: Optional, number of top predictions to display

        Return:
            Matplotlib figure
        """
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
            barlabels = [
                f"{lbl[:12]}..." if len(lbl) > 15 else lbl for lbl in barlabels
            ]

            ax2.set_ylim([0, 1.1])
            ax2.set_xticks(range(top_k))
            ax2.set_xticklabels(barlabels, rotation="vertical")
            ax2.set_ylabel("Probability")

            figure.subplots_adjust(bottom=0.2, wspace=0.3)
            return figure
