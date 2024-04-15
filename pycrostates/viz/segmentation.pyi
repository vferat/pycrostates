from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from mne import BaseEpochs
from mne.io import BaseRaw

from .._typing import ScalarFloatArray as ScalarFloatArray
from .._typing import ScalarIntArray as ScalarIntArray
from ..utils._checks import _check_type as _check_type
from ..utils._checks import _ensure_valid_show as _ensure_valid_show
from ..utils._docs import fill_doc as fill_doc
from ..utils._logs import logger as logger

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
    labels : array of shape ``(n_samples,)``
        Microstates labels attributed to each sample, i.e. the segmentation.
    raw : Raw
        MNE `~mne.io.Raw` instance.
    n_clusters : int
        The number of clusters, i.e. the number of microstates.
    cluster_names : list | None
        Name of the clusters.
    tmin : float
        Start time of the raw data to use in seconds (must be >= 0).
    tmax : float
        End time of the raw data to use in seconds (cannot exceed data duration).
    cmap : str | colormap | None
        The colormap to use. If None, ``viridis`` is used.
    axes : Axes | None
        Either ``None`` to create a new figure or axes on which the segmentation is
        plotted.
    cbar_axes : Axes | None
        Axes on which to draw the colorbar, otherwise the colormap takes space from the main
        axes.
    block : bool
        Whether to halt program execution until the figure is closed.
    show : bool | None
        If True, the figure is shown. If None, the figure is shown if the matplotlib backend
        is interactive.
    verbose : int | str | bool | None
        Sets the verbosity level. The verbosity increases gradually between ``"CRITICAL"``,
        ``"ERROR"``, ``"WARNING"``, ``"INFO"`` and ``"DEBUG"``. If None is provided, the
        verbosity is set to ``"WARNING"``. If a bool is provided, the verbosity is set to
        ``"WARNING"`` for False and to ``"INFO"`` for True.
    **kwargs
        Kwargs are passed to ``axes.plot``.

    Returns
    -------
    fig : Figure
        Matplotlib figure(s) on which topographic maps are plotted.
    """

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
    labels : array of shape ``(n_epochs, n_samples)``
        Microstates labels attributed to each sample, i.e. the segmentation.
    epochs : Epochs
        MNE `~mne.Epochs` instance.
    n_clusters : int
        The number of clusters, i.e. the number of microstates.
    cluster_names : list | None
        Name of the clusters.
    cmap : str | colormap | None
        The colormap to use. If None, ``viridis`` is used.
    axes : Axes | None
        Either ``None`` to create a new figure or axes on which the segmentation is
        plotted.
    cbar_axes : Axes | None
        Axes on which to draw the colorbar, otherwise the colormap takes space from the main
        axes.
    block : bool
        Whether to halt program execution until the figure is closed.
    show : bool | None
        If True, the figure is shown. If None, the figure is shown if the matplotlib backend
        is interactive.
    verbose : int | str | bool | None
        Sets the verbosity level. The verbosity increases gradually between ``"CRITICAL"``,
        ``"ERROR"``, ``"WARNING"``, ``"INFO"`` and ``"DEBUG"``. If None is provided, the
        verbosity is set to ``"WARNING"``. If a bool is provided, the verbosity is set to
        ``"WARNING"`` for False and to ``"INFO"`` for True.
    **kwargs
        Kwargs are passed to ``axes.plot``.

    Returns
    -------
    fig : Figure
        Matplotlib figure on which topographic maps are plotted.
    """

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

def _compatibility_cmap(cmap: str | colors.Colormap | None, n_colors: int):
    """Convert the 'cmap' argument to a colormap.

    Matplotlib 3.6 introduced a deprecation of plt.cm.get_cmap().
    When support for the 3.6 version is dropped, this checker can be removed.
    """
