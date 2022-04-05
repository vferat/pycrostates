from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from mne.io import Info
from mne.viz import plot_topomap
import numpy as  np

from ..utils._checks import _check_type, _check_axes
from ..utils._logs import logger


def plot_cluster_centers(
        cluster_centers,
        info,
        cluster_names=None,
        axes=None,
        block=False,
        **kwargs,
        ):
    """
    Create topographic maps for cluster centers.

    Returns
    -------
    fig : Figure
        Matplotlib figure(s) on which topographic maps are plotted.
    """
    _check_type(cluster_centers, (np.ndarray, ), 'cluster_centers')
    _check_type(info, (Info, ), 'info')
    _check_type(cluster_names, (None, list, tuple), 'cluster_names')
    if axes is not None:
        _check_axes(axes)
    _check_type(block, (bool, ), 'block')

    # check cluster_names
    if cluster_names is None:
        cluster_names = [
            str(k) for k in range(1, cluster_centers.shape[0] + 1)]
    if len(cluster_names) != cluster_centers.shape[0]:
        raise ValueError(
            "Argument 'cluster_centers' and 'cluster_names' should have the "
            "same number of elements.")

    # create axes if needed, and retrieve figure
    n_clusters = cluster_centers.shape[0]
    if axes is None:
        f, axes = plt.subplots(1, n_clusters)
        if isinstance(axes, Axes):
            axes = np.array([axes])  # wrap in an array-like
        # sanity-check
        assert axes.ndim == 1
        # axes formatting
        for ax in axes:
            ax.set_axis_off()
    else:
        # make sure we have enough ax to plot
        if isinstance(axes, Axes) and n_clusters != 1:
            raise ValueError(
                "Argument 'cluster_centers' and 'axes' must contain the same "
                "number of clusters and Axes.")
        elif axes.size != n_clusters:
            raise ValueError(
                "Argument 'cluster_centers' and 'axes' must contain the same "
                "number of clusters and Axes.")
        figs = [ax.get_figure() for ax in axes]
        if len(set(figs)) == 1:
            f = figs[0]
        else:
            f = figs
        del figs

    # remove axes and show from kwargs, issue warning if they are present
    if 'show' in kwargs:
        logger.warning("Argument 'show' can not be provided as kwargs.")
    kwargs = {key: value for key, value in kwargs.items()
              if key not in ('show', )}

    # plot cluster centers
    for k, (center, name) in enumerate(zip(cluster_centers, cluster_names)):
        # select axes from ax
        if axes.ndim == 1:
            ax = axes[k]
        else:
            i = k // axes.shape[1]
            j = k % axes.shape[1]
            ax = axes[i, j]
        # plot
        plot_topomap(center, info, axes=ax, show=False, **kwargs)
        ax.set_title(name)

    plt.show(block=block)
    return f
