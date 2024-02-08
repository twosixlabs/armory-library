import re
from typing import TYPE_CHECKING, List, Optional

from rich.console import Console
from rich.table import Table

import armory.report.common as common

if TYPE_CHECKING:
    import argparse


def configure_args(parser: "argparse.ArgumentParser"):
    parser.description = "Write report to console"
    parser.add_argument(
        "--experiment",
        help="ID of the MLFlow experiment for which to report",
    )
    parser.add_argument(
        "--metrics",
        default=[".*"],
        help="Pattern(s) for metrics to be included in report",
        nargs="*",
    )
    parser.add_argument(
        "--metrics-precision",
        default=3,
        help="Decimal precision for metrics",
        type=int,
    )
    parser.add_argument(
        "--params",
        default=[],
        dest="parameters",
        help="Pattern(s) for parameters to be included in report",
        nargs="*",
    )


def get_matching_run_data(data, data_key: str, patterns: List[str]) -> List[str]:
    keys = set()
    for run in data["runs"]:
        keys.update(run["data"][data_key].keys())

    keys = [
        key for key in keys if any([re.match(pattern, key) for pattern in patterns])
    ]
    return sorted(keys)


def create_multirun_table(
    data, metrics: List[str], metrics_precision: int, parameters: List[str]
):
    metric_keys = get_matching_run_data(data, "metrics", metrics)
    param_keys = get_matching_run_data(data, "params", parameters)

    table = Table(title=data["experiment"]["name"])
    table.add_column("Run", no_wrap=True)
    for key in metric_keys:
        table.add_column(key, style="green")
    for key in param_keys:
        table.add_column(key, style="magenta")

    for run in data["runs"]:
        cols = [run["info"]["run_name"]]
        for key in metric_keys:
            metric = run["data"]["metrics"].get(key)
            if metric is not None:
                cols.append(f"{metric:.{metrics_precision}}")
            else:
                cols.append("")
        for key in param_keys:
            param = run["data"]["params"].get(key)
            if param is not None:
                cols.append(str(param))
            else:
                cols.append("")
        table.add_row(*cols)

    return table


def generate(
    experiment: Optional[str],
    metrics: Optional[List[str]] = None,
    metrics_precision: int = 3,
    parameters: Optional[List[str]] = None,
    **kwargs,
):
    if experiment:
        data = common.dump_experiment(experiment)
    else:
        raise RuntimeError("No experiment or runs provided. Unable to generate report.")

    console = Console()

    if len(data["runs"]) > 0:
        console.print(
            create_multirun_table(
                data,
                metrics=metrics or [],
                metrics_precision=metrics_precision,
                parameters=parameters or [],
            )
        )
