from typing import Optional, Union

from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from mne import BaseEpochs
from mne.io import BaseRaw
from numpy.typing import NDArray

from ..utils._checks import _check_type as _check_type
from ..utils._checks import _ensure_valid_show as _ensure_valid_show
from ..utils._docs import fill_doc as fill_doc
from ..utils._logs import logger as logger

def plot_raw_segmentation(
    labels: NDArray[int],
    raw: BaseRaw,
    n_clusters: int,
    cluster_names: list[str] = None,
    tmin: Optional[Union[int, float]] = None,
    tmax: Optional[Union[int, float]] = None,
    cmap: Optional[str] = None,
    axes: Optional[Axes] = None,
    cbar_axes: Optional[Axes] = None,
    *,
    block: bool = False,
    show: Optional[bool] = None,
    verbose: Optional[str] = None,
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
    labels: NDArray[int],
    epochs: BaseEpochs,
    n_clusters: int,
    cluster_names: list[str] = None,
    cmap: Optional[str] = None,
    axes: Optional[Axes] = None,
    cbar_axes: Optional[Axes] = None,
    *,
    block: bool = False,
    show: Optional[bool] = None,
    verbose: Optional[str] = None,
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
    labels: NDArray[int],
    gfp: NDArray[float],
    times: NDArray[float],
    n_clusters: int,
    cluster_names: list[str] = None,
    cmap: Optional[Union[str, colors.Colormap]] = None,
    axes: Optional[Axes] = None,
    cbar_axes: Optional[Axes] = None,
    *,
    verbose: Optional[str] = None,
    **kwargs,
) -> tuple[plt.Figure, Axes]:
    """Code snippet to plot segmentation for raw and epochs."""

def _compatibility_cmap(cmap: Optional[Union[str, colors.Colormap]], n_colors: int):
    """Convert the 'cmap' argument to a colormap.

    Matplotlib 3.6 introduced a deprecation of plt.cm.get_cmap().
    When support for the 3.6 version is dropped, this checker can be removed.
    """
