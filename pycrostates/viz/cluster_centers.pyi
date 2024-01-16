from typing import Any, Optional, Union

from matplotlib.axes import Axes
from mne import Info
from numpy.typing import NDArray

from .._typing import CHInfo as CHInfo
from ..utils._checks import _check_axes as _check_axes
from ..utils._checks import _check_type as _check_type
from ..utils._docs import fill_doc as fill_doc
from ..utils._logs import logger as logger

_GRADIENT_KWARGS_DEFAULTS: dict[str, str]

def plot_cluster_centers(cluster_centers: NDArray[float], info: Union[Info, CHInfo], cluster_names: list[str]=None, axes: Optional[Union[Axes, NDArray[Axes]]]=None, show_gradient: Optional[bool]=False, gradient_kwargs: dict[str, Any]=..., *, block: bool=False, verbose: Optional[str]=None, **kwargs):
    """Create topographic maps for cluster centers.

    Parameters
    ----------
    cluster_centers : array (n_clusters, n_channels)
        Fitted clusters, i.e. the microstates maps.
    info : Info | ChInfo
        Info instance with a montage used to plot the topographic maps.
    cluster_names : list | None
        Name of the clusters.
    axes : Axes | None
        Either ``None`` to create a new figure or axes (or an array of axes) on which the
        topographic map should be plotted. If the number of microstates maps to plot is
        ``≥ 1``, an array of axes of size ``n_clusters`` should be provided.
    show_gradient : bool
        If True, plot a line between channel locations with highest and lowest values.
    gradient_kwargs : dict
        Additional keyword arguments passed to :meth:`matplotlib.axes.Axes.plot` to plot
        gradient line.
    block : bool
        Whether to halt program execution until the figure is closed.
    verbose : int | str | bool | None
        Sets the verbosity level. The verbosity increases gradually between ``"CRITICAL"``,
        ``"ERROR"``, ``"WARNING"``, ``"INFO"`` and ``"DEBUG"``. If None is provided, the
        verbosity is set to ``"WARNING"``. If a bool is provided, the verbosity is set to
        ``"WARNING"`` for False and to ``"INFO"`` for True.
    **kwargs
        Additional keyword arguments are passed to :func:`mne.viz.plot_topomap`.

    Returns
    -------
    fig : Figure
        Matplotlib figure(s) on which topographic maps are plotted.
    """