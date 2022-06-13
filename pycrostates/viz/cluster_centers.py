"""Visualisation module for plotting cluster centers."""

from typing import List, Optional, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from mne.io import Info
from mne.viz import plot_topomap
from numpy.typing import NDArray

from .._typing import CHInfo
from ..utils._checks import _check_axes, _check_type
from ..utils._docs import fill_doc


@fill_doc
def plot_cluster_centers(
    cluster_centers: NDArray[float],
    info: Union[Info, CHInfo],
    cluster_names: List[str] = None,
    axes: Optional[Axes] = None,
    *,
    block: bool = False,
    **kwargs,
):
    """Create topographic maps for cluster centers.

    Parameters
    ----------
    %(cluster_centers)s
    info : Info | ChInfo
        Info instance with a montage used to plot the topographic maps.
    %(cluster_names)s
    %(axes_topo)s
    %(block)s

    Returns
    -------
    fig : Figure
        Matplotlib figure(s) on which topographic maps are plotted.
    """
    from ..io import ChInfo

    _check_type(cluster_centers, (np.ndarray,), "cluster_centers")
    _check_type(info, (Info, ChInfo), "info")
    _check_type(cluster_names, (None, list, tuple), "cluster_names")
    if axes is not None:
        _check_axes(axes)
    _check_type(block, (bool,), "block")

    # check cluster_names
    if cluster_names is None:
        cluster_names = [
            str(k) for k in range(1, cluster_centers.shape[0] + 1)
        ]
    if len(cluster_names) != cluster_centers.shape[0]:
        raise ValueError(
            "Argument 'cluster_centers' and 'cluster_names' should have the "
            "same number of elements."
        )

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
                f"number of clusters and Axes. Provided: {n_clusters} "
                "microstates maps and only 1 axes."
            )
        elif axes.size != n_clusters:
            raise ValueError(
                "Argument 'cluster_centers' and 'axes' must contain the same "
                f"number of clusters and Axes. Provided: {n_clusters} "
                "microstates maps and {axes.size} axes."
            )
        figs = [ax.get_figure() for ax in axes.flatten()]
        if len(set(figs)) == 1:
            f = figs[0]
        else:
            f = figs
        del figs

    # remove show from kwargs passed to topoplot
    if "show" in kwargs:
        show = kwargs["show"]
        _check_type(show, (bool,), "show")
        del kwargs["show"]
    else:
        show = True

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

    if show:
        plt.show(block=block)
    return f
