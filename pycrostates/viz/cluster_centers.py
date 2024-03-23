"""Visualization module for plotting cluster centers."""

from __future__ import annotations  # c.f. PEP 563, PEP 649

from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from mne import Info
from mne.channels.layout import _find_topomap_coords
from mne.viz import plot_topomap

from ..utils._checks import _check_axes, _check_type, _ensure_valid_show
from ..utils._docs import fill_doc
from ..utils._logs import logger, verbose

if TYPE_CHECKING:
    from typing import Any, Optional, Union

    from .._typing import AxesArray, ScalarFloatArray
    from ..io import ChInfo


_GRADIENT_KWARGS_DEFAULTS: dict[str, str] = {
    "color": "black",
    "linestyle": "-",
    "marker": "P",
}


@fill_doc
@verbose
def plot_cluster_centers(
    cluster_centers: ScalarFloatArray,
    info: Union[Info, ChInfo],
    cluster_names: list[str] = None,
    axes: Optional[Union[Axes, AxesArray]] = None,
    show_gradient: Optional[bool] = False,
    gradient_kwargs: dict[str, Any] = _GRADIENT_KWARGS_DEFAULTS,
    *,
    block: bool = False,
    show: Optional[bool] = None,
    verbose: Optional[str] = None,
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
    show_gradient : bool
        If True, plot a line between channel locations with highest and lowest values.
    gradient_kwargs : dict
        Additional keyword arguments passed to :meth:`matplotlib.axes.Axes.plot` to plot
        gradient line.
    %(block)s
    %(show)s
    %(verbose)s
    **kwargs
        Additional keyword arguments are passed to :func:`mne.viz.plot_topomap`.

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
    _check_type(show_gradient, (bool,), "show_gradient")
    _check_type(
        gradient_kwargs,
        (dict,),
        "gradient_kwargs",
    )
    if gradient_kwargs != _GRADIENT_KWARGS_DEFAULTS and not show_gradient:
        logger.warning(
            "The argument 'gradient_kwargs' has not effect when the argument "
            "'show_gradient' is set to False."
        )
    _check_type(block, (bool,), "block")
    show = _ensure_valid_show(show)

    # check cluster_names
    if cluster_names is None:
        cluster_names = [str(k) for k in range(1, cluster_centers.shape[0] + 1)]
    if len(cluster_names) != cluster_centers.shape[0]:
        raise ValueError(
            "Argument 'cluster_centers' and 'cluster_names' should have the same "
            "number of elements."
        )

    # create axes if needed, and retrieve figure
    n_clusters = cluster_centers.shape[0]
    if axes is None:
        f, axes = plt.subplots(
            1, n_clusters, figsize=(3 * n_clusters, 3), layout="constrained"
        )
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
                "Argument 'cluster_centers' and 'axes' must contain the "
                f"same number of clusters and Axes. Provided: {n_clusters} "
                "microstates maps and only 1 axes."
            )
        elif axes.size != n_clusters:
            raise ValueError(
                "Argument 'cluster_centers' and 'axes' must contain the same "
                f"number of clusters and Axes. Provided: {n_clusters} "
                f"microstates maps and {axes.size} axes."
            )
        figs = [ax.get_figure() for ax in axes.flatten()]
        if len(set(figs)) == 1:
            f = figs[0]
        else:
            f = figs
        del figs

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
        # Add min max vector
        if show_gradient:
            i_min = np.argmin(center)
            i_max = np.argmax(center)
            pos = _find_topomap_coords(info, picks="all")
            ax.plot(
                [pos[i_min, 0], pos[i_max, 0]],
                [pos[i_min, 1], pos[i_max, 1]],
                **gradient_kwargs,
            )
        ax.set_title(name)

    if show:
        plt.show(block=block)
    return f
