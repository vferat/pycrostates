from typing import Tuple, Union

import mne
import numpy as np
from mne import Evoked
from mne.io import BaseRaw
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

from ..utils._checks import _check_type


def plot_cluster_centers(cluster_centers, info, names, block):
    n_clusters = len(cluster_centers)
    fig, axs = plt.subplots(1, n_clusters)
    for c, center in enumerate(cluster_centers):
        mne.viz.plot_topomap(center, info, axes=axs[c], show=False)
        axs[c].set_title(names[c])
    plt.axis('off')
    plt.show(block=block)
    return fig, axs


def plot_segmentation(
        labels: np.ndarray,
        inst: Union[BaseRaw, Evoked],
        cluster_centers: np.ndarray,
        names: list = None,
        tmin: float = 0.0, tmax: float = None) -> Tuple[Figure, Axes]:
    _check_type(inst, (BaseRaw, Evoked))
    inst = inst.copy()
    inst.crop(tmin=tmin, tmax=tmax)
    if isinstance(inst, BaseRaw):
        data = inst.get_data()
    elif isinstance(inst, Evoked):
        data = inst.data  # TODO: I think it supports .get_data() nowadays
    gfp = np.std(data, axis=0)
    times = inst.times + tmin

    n_colors = 1 + len(cluster_centers)
    state_labels = [-1] + list(range(len(cluster_centers)))
    if not names:
        names = ['unlabeled'] + [str(k) for k in range(len(cluster_centers))]
    else:
        names = ['unlabeled'] + names

    labels = labels[(times * inst.info['sfreq']).astype(int)]
    cmap = plt.cm.get_cmap('plasma', n_colors)

    fig = plt.figure(figsize=(10, 4))
    ax = plt.plot(times, gfp, color='black', linewidth=0.2)
    for state, color in zip(state_labels, cmap.colors):
        w = np.where(labels[1:] == state)
        a = np.sort(np.append(w,  np.add(w, 1)))
        x = np.zeros(labels.shape)
        x[a] = 1
        x = x.astype(bool)
        plt.fill_between(times, gfp, color=color, where=x, step=None,
                         interpolate=False)
    norm = colors.Normalize(vmin=0, vmax=n_colors)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ticks=[i + 0.5 for i in range(n_colors)])
    cbar.ax.set(yticklabels=names)
    plt.xlabel('Time (s)')
    plt.title('Segmentation')
    plt.autoscale(tight=True)
    plt.show()
    return fig, ax
