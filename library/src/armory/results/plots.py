from typing import Optional, Sequence

from armory.results.results import BatchExports


def plot_batches(*batches: BatchExports, titles: Optional[Sequence[str]] = None):
    import matplotlib.pyplot as plt

    with plt.ioff():
        figure = plt.figure()
        subfigures = figure.subfigures(nrows=1, ncols=len(batches))

        for batch_idx, batch in enumerate(batches):
            subfig = subfigures[batch_idx]
            batch.plot("objects.png", figure=subfig)

            if titles is not None and batch_idx < len(titles):
                subfig.suptitle(titles[batch_idx])

        return figure
