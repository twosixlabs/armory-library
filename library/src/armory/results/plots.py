import io
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

import PIL.Image

from armory.results.utils import get_next_dash_port

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


# def plot_batches(
#     *batches: "BatchExports",
#     filename: Optional[str] = None,
#     max_samples: Optional[int] = None,
#     samples: Optional[Sequence[int]] = None,
#     titles: Optional[Sequence[str]] = None,
# ):
#     import matplotlib.pyplot as plt

#     with plt.ioff():
#         figure = plt.figure()
#         subfigures = figure.subfigures(nrows=1, ncols=len(batches))

#         for batch_idx, batch in enumerate(batches):
#             subfig = subfigures[batch_idx]
#             batch.plot(
#                 filename=filename,
#                 figure=subfig,
#                 max_samples=max_samples,
#                 samples=samples,
#             )

#             if titles is not None and batch_idx < len(titles):
#                 subfig.suptitle(titles[batch_idx])

#         return figure


# def plot_samples(
#     *batches: "BatchExports",
#     samples: Sequence[int],
#     to_plot: Callable[["SampleExports"], "Plottable"],
#     titles: Optional[Sequence[str]] = None,
#     **kwargs,
# ):
#     import matplotlib.pyplot as plt

#     with plt.ioff():
#         figure = plt.figure()
#         subfigures = figure.subfigures(nrows=len(samples), ncols=len(batches))

#         for sample_idx, sample_num in enumerate(samples):
#             for batch_idx, batch in enumerate(batches):
#                 subfig = subfigures[sample_idx][batch_idx]
#                 to_plot(batch.sample(sample_num)).plot(figure=subfig, **kwargs)

#                 if sample_idx == 0 and titles is not None and batch_idx < len(titles):
#                     subfig.suptitle(titles[batch_idx])

#         return figure


def plot_metrics(
    *runs: "EvaluationResults",
    blacklist: Optional[Sequence[str]] = None,
    dark: bool = False,
    debug: bool = False,
    height: int = 400,
    metric_label: str = "Metric",
    port: Optional[int] = None,
    precision: int = 3,
    title: str = "Metrics",
    whitelist: Optional[Sequence[str]] = None,
):
    import dash

    columns = [
        {"id": "metric_key", "name": metric_label},
    ]
    metric_keys = set()
    for run in runs:
        metric_keys.update(run.metrics.keys())
        columns.append({"id": run.run_id, "name": run.run_name})

    if whitelist:
        metric_keys &= set(whitelist)
    if blacklist:
        metric_keys -= set(blacklist)

    def format(v):
        return f"{v:.{precision}f}" if v is not None else ""

    data = []
    for metric_key in sorted(metric_keys):
        row = {"metric_key": metric_key}
        for run in runs:
            row[run.run_id] = format(run.metrics.get(metric_key, None))
        data.append(row)

    app = dash.Dash()
    app.layout = dash.html.Div(
        children=[
            dash.html.H1(
                children=title,
                style={
                    "color": "white" if dark else "black",
                    "textAlign": "center",
                },
            ),
            dash.dash_table.DataTable(
                data=data,
                columns=columns,
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
                    "border": ("1px solid dimgray" if dark else "1px solid lightgray"),
                    "color": "white" if dark else "black",
                },
                style_cell_conditional=[
                    {
                        "if": {"column_id": "metric_key"},
                        "backgroundColor": (
                            "rgb(10, 10, 10)" if dark else "rgb(229, 229, 229)"
                        ),
                        "color": "white" if dark else "black",
                        "fontWeight": "bold",
                        "width": "20%",
                    },
                ],
                style_data_conditional=[
                    {
                        "if": {"column_id": run.run_id, "row_index": "odd"},
                        "backgroundColor": (
                            "rgb(23, 23, 23)" if dark else "rgb(250, 250, 250)"
                        ),
                    }
                    for run in runs
                ],
            ),
        ],
        style={"background": "rgb(30, 30, 30)" if dark else "white"},
    )

    app.run(
        debug=debug,
        jupyter_height=height,
        port=str(port) if port is not None else get_next_dash_port(),
    )


def plot_params(
    *runs: "EvaluationResults",
    blacklist: Optional[Sequence[str]] = None,
    dark: bool = False,
    debug: bool = False,
    height: int = 400,
    hide_same: bool = False,
    highlight_diff: bool = True,
    param_label: str = "Parameter",
    port: Optional[int] = None,
    title: str = "Parameters",
    whitelist: Optional[Sequence[str]] = None,
):
    import dash

    columns = [
        {"id": "param_key", "name": param_label},
    ]
    param_keys = set()
    for run in runs:
        param_keys.update(run.params.keys())
        columns.append({"id": run.run_id, "name": run.run_name})

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
            "true" if len(set([row[run.run_id] for run in runs])) > 1 else "false"
        )
        if hide_same and row["diff_key"] == "false":
            continue
        if not highlight_diff:
            row["diff_key"] = "false"
        data.append(row)

    app = dash.Dash()
    app.layout = dash.html.Div(
        children=[
            dash.html.H1(
                children=title,
                style={
                    "color": "white" if dark else "black",
                    "textAlign": "center",
                },
            ),
            dash.dash_table.DataTable(
                data=data,
                columns=columns,
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
                    "border": ("1px solid dimgray" if dark else "1px solid lightgray"),
                    "color": "white" if dark else "black",
                    "overflow": "hidden",
                    "textAlign": "left",
                    "textOverflow": "ellipsis",
                },
                style_cell_conditional=[
                    {
                        "if": {"column_id": "param_key"},
                        "backgroundColor": (
                            "rgb(10, 10, 10)" if dark else "rgb(229, 229, 229)"
                        ),
                        "color": "white" if dark else "black",
                        "fontWeight": "bold",
                        "width": "20%",
                    }
                ]
                + [
                    {
                        "if": {"column_id": run.run_id},
                        "maxWidth": 0,
                        "textAlign": "left",
                    }
                    for run in runs
                ],
                style_data_conditional=[
                    {
                        "if": {"column_id": run.run_id, "row_index": "odd"},
                        "backgroundColor": (
                            "rgb(23, 23, 23)" if dark else "rgb(250, 250, 250)"
                        ),
                    }
                    for run in runs
                ]
                + [
                    {
                        "if": {"filter_query": "{diff_key} = 'true'"},
                        "backgroundColor": "rgb(74, 222, 128)",
                    }
                ],
            ),
        ],
        style={"background": "rgb(30, 30, 30)" if dark else "white"},
    )

    app.run(
        debug=debug,
        jupyter_height=height,
        port=str(port) if port is not None else get_next_dash_port(),
    )
