import io
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Union

import PIL.Image

if TYPE_CHECKING:
    import matplotlib.figure

    from armory.results.results import EvaluationResults


def _to_pil(
    figure: Union["matplotlib.figure.Figure", PIL.Image.Image]
) -> PIL.Image.Image:
    if isinstance(figure, PIL.Image.Image):
        return figure

    buf = io.BytesIO()
    figure.savefig(buf, bbox_inches="tight")
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img


def plot_in_grid(
    cells: Union[
        Sequence[Union["matplotlib.figure.Figure", PIL.Image.Image]],
        Sequence[Sequence[Union["matplotlib.figure.Figure", PIL.Image.Image]]],
    ],
    border: bool = False,
    columns: Optional[Sequence[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    rows: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    vertical: bool = False,
):
    """
    Create a matplotlib figure with the given cells arranged in a grid.

    Args:
        cells: 1D list or 2D array of matplotlib figures or PIL images to display
        border: If True, draw borders around cell images
        columns: Labels for columns of the grid
        figsize: Size of the figure
        rows: Labels for rows of the grid
        title: Title for the figure
        vertical: If True, arrange 1D cells vertically instead of horizontally

    Return:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    with plt.ioff():
        is_2d = isinstance(cells[0], Sequence)
        if is_2d:
            nrows = len(cells)
            ncols = len(cells[0])
        elif vertical:
            nrows = len(cells)
            ncols = 1
        else:
            nrows = 1
            ncols = len(cells)

        figure = plt.figure(figsize=figsize)
        axes = figure.subplots(nrows=nrows, ncols=ncols)

        for row_idx, row in enumerate(cells):
            if is_2d:
                for col_idx, cell in enumerate(row):
                    ax = axes[row_idx, col_idx]
                    ax.imshow(_to_pil(cell))

                    if row_idx == 0 and columns is not None and col_idx < len(columns):
                        ax.set_title(columns[col_idx])

                    if col_idx == 0 and rows is not None and row_idx < len(rows):
                        ax.set_ylabel(rows[row_idx])
            else:
                ax = axes[row_idx]
                ax.imshow(_to_pil(row))

                if not vertical and columns is not None and row_idx < len(columns):
                    ax.set_title(columns[row_idx])

                if vertical and rows is not None and row_idx < len(rows):
                    ax.set_ylabel(rows[row_idx])

        for ax in figure.axes:
            # Cannot use ax.axis("off") because it will remove any labels, and
            # we only want to remove the borders and tick marks
            if not border:
                for spine in ax.spines.values():
                    spine.set_visible(False)
            ax.grid(False)
            ax.tick_params(
                bottom=False,
                left=False,
                labelbottom=False,
                labelleft=False,
            )

        if title:
            figure.suptitle(title)

        figure.tight_layout()

        return figure


def _tag_with_style(tag: str, style: Dict[str, Any], **attrs) -> str:
    style_str = "; ".join([f"{k}: {v}" for k, v in style.items()])
    attrs_str = " ".join([f"{k}='{v}'" for k, v in attrs.items()])
    return f"<{tag} style='{style_str}' {attrs_str}>"


def plot_metrics(
    *runs: "EvaluationResults",
    blacklist: Optional[Sequence[str]] = None,
    metric_label: str = "Metric",
    precision: int = 3,
    whitelist: Optional[Sequence[str]] = None,
):
    """
    Create an HTML table of metrics from multiple runs.

    Args:
        runs: EvaluationResults from runs to compare
        blacklist: Optional, list of metrics to exclude
        metric_label: Optional, label for the metric column
        precision: Optional, number of decimal places to display for floating
            point values
        whitelist: Optional, list of metrics to include

    Return:
        IPython HTML object for the table
    """
    from IPython.core.display import HTML

    metric_keys = set()
    for run in runs:
        metric_keys.update(run.metrics.keys())

    if whitelist:
        metric_keys &= set(whitelist)
    if blacklist:
        metric_keys -= set(blacklist)

    def format(v):
        return f"{v:.{precision}f}" if v is not None else ""

    table = "<table>"

    table += "<tr>"
    for header in [metric_label] + [run.run_name for run in runs]:
        table += _tag_with_style("th", {"font-weight": "bold", "text-align": "center"})
        table += f"{header}</th>"
    table += "</tr>"

    for metric_key in sorted(metric_keys):
        table += f"<tr><td>{metric_key}</td>"
        for run in runs:
            table += f"<td>{format(run.metrics.get(metric_key, None))}</td>"
        table += "</tr>"

    table += "</table>"
    return HTML(table)


def plot_params(
    *runs: "EvaluationResults",
    blacklist: Optional[Sequence[str]] = None,
    hide_same: bool = False,
    highlight_diff: bool = True,
    param_label: str = "Parameter",
    whitelist: Optional[Sequence[str]] = None,
):
    """
    Create an HTML table of parameters from multiple runs.

    Args:
        runs: EvaluationResults from runs to compare
        blacklist: Optional, list of parameters to exclude
        hide_same: If True, hide parameters that are the same across all runs
        highlight_diff: If True, highlight parameters that are different across
            runs
        param_label: Optional, label for the parameter column
        whitelist: Optional, list of parameters to include

    Return:
        IPython HTML object for the table
    """
    from IPython.core.display import HTML

    param_keys = set()
    for run in runs:
        param_keys.update(run.params.keys())

    if whitelist:
        param_keys &= set(whitelist)
    if blacklist:
        param_keys -= set(blacklist)

    data = []
    for param_key in sorted(param_keys):
        row = {"param_key": param_key}
        for run in runs:
            row[run.run_id] = run.params.get(param_key, "")
        row["diff_key"] = (
            True if len(set([row[run.run_id] for run in runs])) > 1 else False
        )
        if hide_same and not row["diff_key"]:
            continue
        if not highlight_diff:
            row["diff_key"] = False
        data.append(row)

    table = "<table>"

    table += "<tr>"
    for header in [param_label] + [run.run_name for run in runs]:
        table += _tag_with_style("th", {"font-weight": "bold", "text-align": "center"})
        table += f"{header}</th>"
    table += "</tr>"

    for row in data:
        table += "<tr>"
        table += _tag_with_style(
            "td", {"font-weight": "bold", "text-align": "left", "width": "20%"}
        )
        table += f"{row['param_key']}</td>"
        for run in runs:
            style = {
                "max-width": "0",
                "overflow": "hidden",
                "text-align": "left",
                "text-overflow": "ellipsis",
            }
            if row["diff_key"]:
                style["color"] = "#4ade80"
            table += _tag_with_style("td", style, title=row[run.run_id])
            table += f"{row[run.run_id]}</td>"
        table += "</tr>"

    table += "</table>"
    return HTML(table)
