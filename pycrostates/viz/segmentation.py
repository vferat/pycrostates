"""Visualisation module for plotting segmentations."""

from __future__ import annotations  # c.f. PEP 563, PEP 649

from typing import TYPE_CHECKING

import numpy as np
from matplotlib import colormaps, colors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from mne import BaseEpochs
from mne.io import BaseRaw
from mne.utils import check_version

from ..utils._checks import _check_type, _ensure_valid_show
from ..utils._docs import fill_doc
from ..utils._logs import logger, verbose

if TYPE_CHECKING:
    from .._typing import ScalarFloatArray, ScalarIntArray


@fill_doc
def plot_raw_segmentation(
    labels: ScalarIntArray,
    raw: BaseRaw,
    n_clusters: int,
    cluster_names: list[str] = None,
    tmin: int | float | None = None,
    tmax: int | float | None = None,
    cmap: str | None = None,
    axes: Axes | None = None,
    cbar_axes: Axes | None = None,
    *,
    block: bool = False,
    show: bool | None = None,
    verbose: str | None = None,
    **kwargs,
):
    """Plot raw segmentation.

    Parameters
    ----------
    %(labels_raw)s
    raw : Raw
        MNE `~mne.io.Raw` instance.
    %(n_clusters)s
    %(cluster_names)s
    %(tmin_raw)s
    %(tmax_raw)s
    %(cmap)s
    %(axes_seg)s
    %(axes_cbar)s
    %(block)s
    %(show)s
    %(verbose)s
    **kwargs
        Kwargs are passed to ``axes.plot``.

    Returns
    -------
    fig : Figure
        Matplotlib figure(s) on which topographic maps are plotted.
    """
    _check_type(labels, (np.ndarray,), "labels")  # 1D array (n_times, )
    if labels.ndim != 1:
        raise ValueError("Argument 'labels' should be a 1D array.")
    _check_type(raw, (BaseRaw,), "raw")
    _check_type(block, (bool,), "block")
    show = _ensure_valid_show(show)

    data = raw.get_data(tmin=tmin, tmax=tmax)
    gfp = np.std(data, axis=0)
    # build times array instead of using raw.times because the time-based
    # selection in MNE can be a bit funky.
    if tmin is None:
        tmin = raw.times[0]
    times = np.arange(
        tmin,
        tmin + gfp.size / raw.info["sfreq"],
        1 / raw.info["sfreq"],
    )
    labels = labels[(times * raw.info["sfreq"]).astype(int)]

    # make sure shapes are correct
    if data.shape[1] != labels.size:
        raise ValueError(
            "Argument 'labels' and 'raw' do not have the same number of samples."
        )

    fig, axes = _plot_segmentation(
        labels,
        gfp,
        times,
        n_clusters,
        cluster_names,
        cmap,
        axes,
        cbar_axes,
        verbose=verbose,
        **kwargs,
    )

    # format
    axes.set_xlabel("Time (s)")

    if show:
        plt.show(block=block)
    return fig


@fill_doc
def plot_epoch_segmentation(
    labels: ScalarIntArray,
    epochs: BaseEpochs,
    n_clusters: int,
    cluster_names: list[str] = None,
    cmap: str | None = None,
    axes: Axes | None = None,
    cbar_axes: Axes | None = None,
    *,
    block: bool = False,
    show: bool | None = None,
    verbose: str | None = None,
    **kwargs,
):
    """
    Plot epochs segmentation.

    Parameters
    ----------
    %(labels_epo)s
    epochs : Epochs
        MNE `~mne.Epochs` instance.
    %(n_clusters)s
    %(cluster_names)s
    %(cmap)s
    %(axes_seg)s
    %(axes_cbar)s
    %(block)s
    %(show)s
    %(verbose)s
    **kwargs
        Kwargs are passed to ``axes.plot``.

    Returns
    -------
    fig : Figure
        Matplotlib figure on which topographic maps are plotted.
    """
    _check_type(labels, (np.ndarray,), "labels")  # 1D array (n_times, )
    if labels.ndim != 2:
        raise ValueError("Argument labels should be a 2D array.")
    _check_type(epochs, (BaseEpochs,), "epochs")
    _check_type(block, (bool,), "block")
    show = _ensure_valid_show(show)

    kwargs_epochs = dict(copy=False) if check_version("mne", "1.6") else dict()
    data = epochs.get_data(**kwargs_epochs).swapaxes(0, 1)
    data = data.reshape(data.shape[0], -1)
    gfp = np.std(data, axis=0)
    times = np.arange(0, data.shape[-1])
    labels = labels.reshape(-1)

    # make sure shapes are correct
    if data.shape[1] != labels.size:
        raise ValueError(
            "Argument 'labels' and 'epochs' do not have the same number of samples."
        )

    fig, axes = _plot_segmentation(
        labels,
        gfp,
        times,
        n_clusters,
        cluster_names,
        cmap,
        axes,
        cbar_axes,
        verbose=verbose,
        **kwargs,
    )

    # format
    x_ticks = np.linspace(
        epochs.times.size // 2,
        data.shape[-1] - epochs.times.size // 2,
        len(epochs),
    )
    x_tick_labels = [str(i) for i in range(1, len(epochs) + 1)]
    axes.set_xticks(x_ticks, x_tick_labels)
    axes.set_xlabel("Epochs")

    # add epoch lines
    x = np.linspace(
        epochs.times.size,
        data.shape[-1] - epochs.times.size,
        len(epochs) - 1,
    )
    axes.vlines(x, 0, gfp.max(), linestyles="dashed", colors="black")

    if show:
        plt.show(block=block)
    return fig


@verbose
def _plot_segmentation(
    labels: ScalarIntArray,
    gfp: ScalarFloatArray,
    times: ScalarFloatArray,
    n_clusters: int,
    cluster_names: list[str] = None,
    cmap: str | colors.Colormap | None = None,
    axes: Axes | None = None,
    cbar_axes: Axes | None = None,
    *,
    verbose: str | None = None,
    **kwargs,
) -> tuple[plt.Figure, Axes]:
    """Code snippet to plot segmentation for raw and epochs."""
    _check_type(labels, (np.ndarray,), "labels")  # 1D array (n_times, )
    _check_type(gfp, (np.ndarray,), "gfp")  # 1D array (n_times, )
    _check_type(times, (np.ndarray,), "times")  # 1D array (n_times, )
    _check_type(n_clusters, ("int",), "n_clusters")
    if n_clusters <= 0:
        raise ValueError(
            f"Provided number of clusters {n_clusters} is invalid. The number of "
            "clusters must be strictly positive."
        )
    _check_type(cluster_names, (None, list, tuple), "cluster_names")
    _check_type(cmap, (None, str, colors.Colormap), "cmap")
    _check_type(axes, (None, Axes), "ax")
    _check_type(cbar_axes, (None, Axes), "cbar_ax")

    # check cluster_names
    if cluster_names is None:
        cluster_names = [str(k) for k in range(1, n_clusters + 1)]
    if len(cluster_names) != n_clusters:
        raise ValueError(
            "Argument 'cluster_names' should have the 'n_clusters' elements. "
            f"Provided: {len(cluster_names)} names for {n_clusters} clusters."
        )

    if axes is None:
        fig, axes = plt.subplots(1, 1, layout="constrained")
    else:
        fig = axes.get_figure()

    # add color and linewidth if absent from kwargs
    if "color" not in kwargs:
        kwargs["color"] = "black"
    if "linewidth" not in kwargs:
        kwargs["linewidth"] = 0.2

    # define states and colors
    state_labels = [-1] + list(range(n_clusters))
    cluster_names = ["unlabeled"] + cluster_names
    n_colors = n_clusters + 1
    cmap = _compatibility_cmap(cmap, n_colors)

    # plot
    axes.plot(times, gfp, **kwargs)
    for state, color in zip(state_labels, cmap.colors, strict=False):
        pos = np.where(labels == state)[0]
        if len(pos):
            pos = np.unique([pos, pos + 1])
            x = np.zeros(labels.shape).astype(bool)
            if pos[-1] == labels.size:
                pos = pos[:-1]
            x[pos] = True
            axes.fill_between(
                times, gfp, color=color, where=x, step=None, interpolate=False
            )
    logger.info(
        "For visualization purposes, the last segment appears truncated by 1 sample. "
        "In the case where the last segment is 1 sample long, it does not appear."
    )

    # commonm formatting
    axes.set_title("Segmentation")
    axes.autoscale(tight=True)

    # color bar
    norm = colors.Normalize(vmin=0, vmax=n_colors)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    colorbar = plt.colorbar(
        sm, cax=cbar_axes, ax=axes, ticks=[i + 0.5 for i in range(n_colors)]
    )
    colorbar.ax.set_yticklabels(cluster_names)

    return fig, axes


def _compatibility_cmap(
    cmap: str | colors.Colormap | None,
    n_colors: int,
):
    """Convert the 'cmap' argument to a colormap.

    Matplotlib 3.6 introduced a deprecation of plt.cm.get_cmap().
    When support for the 3.6 version is dropped, this checker can be removed.
    """
    if check_version("matplotlib", "3.6"):
        if cmap is None:
            cmap = colormaps["viridis"]
        elif isinstance(cmap, str):
            cmap = colormaps[cmap]  # the cmap name is checked by matplotlib
        cmap = cmap.resampled(n_colors)
    else:
        if isinstance(cmap, (str | type(None))):
            cmap = plt.cm.get_cmap(cmap, n_colors)
        else:
            raise RuntimeError(
                "User-defined colormaps are supported as of matplotlib 3.6 "
                "and above. Please update matplotlib or provide a colormap "
                "name as a string."
            )
    return cmap
