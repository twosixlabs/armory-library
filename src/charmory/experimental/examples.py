import math
import random

from matplotlib import pyplot as plt


def show_samples(dataset, samples=3, n_col=3):
    """shows random selections from dataset with `samples` distinct labels"""

    chosen_labels = set()
    chosen = []
    while len(chosen) < samples:
        choice = random.choice(dataset)
        label = choice["label"]
        if label not in chosen_labels:
            chosen.append(choice)
            chosen_labels.add(label)

    n_row = math.ceil(samples / n_col)
    _, axs = plt.subplots(n_row, n_col)
    axs = axs.flatten()
    for sample, ax in zip(chosen, axs):
        ax.imshow(sample["image"])
    plt.show()
