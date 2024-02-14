import json
import pathlib
from typing import TYPE_CHECKING, List, Optional

import importlib_resources

import armory.report.common as common

if TYPE_CHECKING:
    import argparse

    from importlib_resources.abc import Traversable


BLACKLIST = ["armory-evaluation-data.js", "tsconfig.json", "tailwind.config.js"]


def configure_args(parser: "argparse.ArgumentParser"):
    parser.description = "Produce HTML report for an Armory evaluation"
    parser.add_argument(
        "--out",
        default="./public",
        help="Directory in which to place generated HTML files",
    )
    parser.add_argument(
        "--experiment",
        help="ID of the MLFlow experiment for which to report",
    )
    parser.add_argument(
        "--baseline-chain",
        help="Perturbation chain to which to compare all other chains",
    )
    parser.add_argument(
        "--baseline-run",
        help="Run ID to which to compare all other runs",
    )
    parser.add_argument(
        "--hide-chains",
        help="Perturbation chains to be hidden by default in the report",
        nargs="*",
    )
    parser.add_argument(
        "--hide-metrics",
        help="Metrics to be hidden by default in the report",
        nargs="*",
    )
    parser.add_argument(
        "--show-parameters",
        help="Parameters to be shown by default in the report",
        nargs="*",
    )
    parser.add_argument(
        "--metric-precision",
        default=3,
        help="Default decimal precision for metrics",
        type=int,
    )
    parser.add_argument(
        "--metric-types",
        help="Type of metric (high or low is better) in the form of ':'-separated key-value pairs (e.g., 'accuracy:high')",
        nargs="*",
    )
    parser.add_argument(
        "--export-batches",
        help="Batches from which to include exported samples",
        nargs="*",
    )
    parser.add_argument(
        "--max-samples",
        help="Maximum number of samples to include from each batch",
        type=int,
    )


def copy_resources(srcdir: "Traversable", outdir: pathlib.Path):
    for entry in srcdir.iterdir():
        if entry.name in BLACKLIST:
            continue
        if entry.is_dir():
            subdir = outdir / entry.name
            subdir.mkdir(parents=True, exist_ok=True)
            copy_resources(entry, subdir)
        else:
            with open(outdir / entry.name, "wb") as outfile:
                outfile.write(entry.read_bytes())


def generate(
    out: str,
    experiment: Optional[str],
    baseline_chain: Optional[str],
    baseline_run: Optional[str],
    hide_chains: List[str],
    hide_metrics: List[str],
    show_parameters: List[str],
    metric_precision: int,
    metric_types: List[str],
    export_batches: List[str],
    max_samples: Optional[int],
    **kwargs,
):
    outpath = pathlib.Path(out)
    outpath.mkdir(parents=True, exist_ok=True)

    if experiment:
        data = common.dump_experiment(experiment)
        data["settings"] = dict(
            baseline_chain=baseline_chain,
            baseline_run=baseline_run,
            hide_chains=hide_chains,
            hide_metrics=hide_metrics,
            show_parameters=show_parameters,
            metric_precision=metric_precision,
            metric_types={
                kv[0]: kv[1] for kv in [kv.split(":") for kv in metric_types]
            },
        )
        if export_batches:
            for run in data["runs"]:
                run["artifacts"] = common.dump_artifacts(
                    run_id=run["info"]["run_id"],
                    batches=export_batches,
                    max_samples=max_samples,
                    extension="png",
                    outdir=outpath / "assets/img" / run["info"]["run_id"],
                )

        with open(outpath / "armory-evaluation-data.js", "w") as outfile:
            jsdata = json.dumps(data, indent=2, sort_keys=True)
            outfile.write(f"export default {jsdata};")
    else:
        raise RuntimeError("No experiment or runs provided. Unable to generate report.")

    print(f"Producing HTML output in {out}...")
    copy_resources(importlib_resources.files(__package__) / "www", outpath)
    print("Done")
