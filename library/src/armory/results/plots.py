from typing import Optional, Sequence

from armory.results.results import BatchExports, EvaluationResults
from armory.results.utils import get_next_dash_port


def plot_batches(
    *batches: BatchExports,
    filename: Optional[str] = None,
    max_samples: Optional[int] = None,
    titles: Optional[Sequence[str]] = None,
):
    import matplotlib.pyplot as plt

    with plt.ioff():
        figure = plt.figure()
        subfigures = figure.subfigures(nrows=1, ncols=len(batches))

        for batch_idx, batch in enumerate(batches):
            subfig = subfigures[batch_idx]
            batch.plot(filename=filename, figure=subfig, max_samples=max_samples)

            if titles is not None and batch_idx < len(titles):
                subfig.suptitle(titles[batch_idx])

        return figure


def plot_metrics(
    *runs: EvaluationResults,
    dark: bool = False,
    debug: bool = False,
    height: int = 400,
    metric_label: str = "Metric",
    port: Optional[int] = None,
    precision: int = 3,
    title: str = "Metrics",
):
    import dash

    columns = [
        {"id": "metric_key", "name": metric_label},
    ]
    metric_keys = set()
    for run in runs:
        metric_keys.update(run.metrics.keys())
        columns.append({"id": run.run_id, "name": run.run_name})

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
