from typing import Optional, Sequence

from armory.results.results import BatchExports


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
