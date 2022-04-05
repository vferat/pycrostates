from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from mne.io import Info
from mne.viz import plot_topomap
import numpy as  np

from ..utils._checks import _check_type, _check_ax


def plot_cluster_centers(
        cluster_centers,
        info,
        cluster_names=None,
        ax=None,
        block=False,
        **kwargs,
        ):
    """
    Create topographic plot for cluster_centers.

    Parameters
    ----------
    cluster_centers : TYPE
        DESCRIPTION.
    info : TYPE
        DESCRIPTION.
    cluster_names : TYPE, optional
        DESCRIPTION. The default is None.
    ax : TYPE, optional
        DESCRIPTION. The default is None.
    block : TYPE, optional
        DESCRIPTION. The default is False.
    **kwargs : Extra keyword arguments are passed to
        :func:`mne.viz.plot_topomap`.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    """
    _check_type(cluster_centers, (np.ndarray, ), 'cluster_centers')
    _check_type(info, (Info, ), 'info')
    _check_type(cluster_names, (None, list, tuple), 'cluster_names')
    if ax is not None:
        _check_ax(ax)
    _check_type(block, (bool, ), 'block')

    # check cluster_names
    if cluster_names is None:
        cluster_names = [str(k) for k in range(1, cluster_centers.size + 1)]
    if len(cluster_names) != cluster_centers.size:
        raise ValueError(
            "Argument 'cluster_centers' and 'cluste_names' should have the "
            "same number of elements.")

    # create axes if needed, and retrieve figure
    n_clusters = cluster_centers.shape[0]
    if ax is None:
        f, ax = plt.subplots(1, n_clusters)
        if isinstance(ax, Axes):
            ax = np.array([ax])  # wrap in an array-like
        # sanity-check
        assert ax.ndim == 1
        # axes formatting
        for a in ax:
            a.set_axis_off()
    else:
        # make sure we have enough ax to plot
        if isinstance(ax, Axes) and n_clusters != 1:
            raise ValueError(
                "Argument 'cluster_centers' and 'ax' must contain the same "
                "number of clusters and Axes.")
        elif ax.size != n_clusters:
            raise ValueError(
                "Argument 'cluster_centers' and 'ax' must contain the same "
                "number of clusters and Axes.")
        figs = [a.get_figure() for a in ax]
        if len(set(figs)) == 1:
            f = figs[0]
        else:
            f = figs
        del figs

    # removea axes and show from kwargs
    kwargs = {key: value for key, value in kwargs.items()
              if key not in ('axes', 'show')}

    # plot cluster centers
    for k, (center, name) in enumerate(zip(cluster_centers, cluster_names)):
        # select axes from ax
        if ax.ndim == 1:
            axes = ax[k]
        else:
            i = k // ax.shape[1]
            j = k % ax.shape[1]
            axes = ax[i, j]
        # plot
        plot_topomap(center, info, axes=axes, show=False, **kwargs)
        axes.set_title(name)

    plt.show(block=block)
    return f
