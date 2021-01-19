from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import mne
import numpy as np


def plot_segmentation(
    labels: np.ndarray,
    raw: mne.io.RawArray,
    names: list = None,
    tmin: float = 0.0,
    tmax: float = None,
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """[summary]

    Args:
        labels (np.ndarray): [description]
        raw (mne.io.RawArray): [description]
        names (list, optional): [description]. Defaults to None.
        tmin (float, optional): [description]. Defaults to 0.0.
        tmax (float, optional): [description]. Defaults to None.

    Returns:
        Tuple[mpl.figure.Figure, mpl.axes.Axes]: [description]
    """

    raw_ = raw.copy()
    raw_ = raw_.crop(tmin=tmin, tmax=tmax)
    gfp = np.std(raw_.get_data(), axis=0)
    times = raw_.times + tmin
    n_states = np.max(labels) + 1
    if names is None:
        names = ["unlabeled"]
        names.extend([f"Microstate {i+1}" for i in range(n_states - 1)])
    labels = labels[(times * raw_.info["sfreq"]).astype(int)]
    cmap = plt.cm.get_cmap("plasma", n_states)

    fig = plt.figure(figsize=(10, 4))
    ax = plt.plot(times, gfp, color="black", linewidth=0.2)
    for state, color in zip(range(n_states), cmap.colors):
        w = np.where(labels[1:] == state)
        a = np.sort(np.append(w, np.add(w, 1)))
        x = np.zeros(labels.shape)
        x[a] = 1
        x = x.astype(bool)
        plt.fill_between(times, gfp, color=color, where=x, step=None, interpolate=False)
    norm = mpl.colors.Normalize(vmin=0, vmax=n_states)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ticks=[i + 0.5 for i in range(n_states)])
    cbar.ax.set(yticklabels=names)
    plt.xlabel("Time (s)")
    plt.title("Segmentation")
    plt.autoscale(tight=True)
    plt.show()
    return (fig, ax)
