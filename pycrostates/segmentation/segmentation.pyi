from _typeshed import Incomplete
from matplotlib.axes import Axes as Axes
from mne import BaseEpochs
from mne.io import BaseRaw

from ..utils._checks import _check_type as _check_type
from ..utils._docs import fill_doc as fill_doc
from ..viz import plot_epoch_segmentation as plot_epoch_segmentation
from ..viz import plot_raw_segmentation as plot_raw_segmentation
from ._base import _BaseSegmentation as _BaseSegmentation

class RawSegmentation(_BaseSegmentation):
    """
    Contains the segmentation of a `~mne.io.Raw` instance.

    Parameters
    ----------
    labels : array of shape ``(n_samples,)``
        Microstates labels attributed to each sample, i.e. the segmentation.
    raw : Raw
        `~mne.io.Raw` instance used for prediction.
    cluster_centers : array (n_clusters, n_channels)
         Clusters, i.e. the microstates maps used to compute the segmentation.
    cluster_names : list | None
        Name of the clusters.
    predict_parameters : dict | None
        The prediction parameters.
    """

    def __init__(self, *args, **kwargs) -> None: ...
    def plot(
        self,
        tmin: int | float | None = None,
        tmax: int | float | None = None,
        cmap: str | None = None,
        axes: Axes | None = None,
        cbar_axes: Axes | None = None,
        *,
        block: bool = False,
        show: bool | None = None,
        verbose: str | None = None,
    ):
        """Plot the segmentation.

        Parameters
        ----------
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

        Returns
        -------
        fig : Figure
            Matplotlib figure containing the segmentation.
        """

    @property
    def raw(self) -> BaseRaw:
        """`~mne.io.Raw` instance from which the segmentation was computed."""

class EpochsSegmentation(_BaseSegmentation):
    """Contains the segmentation of an `~mne.Epochs` instance.

    Parameters
    ----------
    labels : array of shape ``(n_epochs, n_samples)``
        Microstates labels attributed to each sample, i.e. the segmentation.
    epochs : Epochs
        `~mne.Epochs` instance used for prediction.
    cluster_centers : array (n_clusters, n_channels)
         Clusters, i.e. the microstates maps used to compute the segmentation.
    cluster_names : list | None
        Name of the clusters.
    predict_parameters : dict | None
        The prediction parameters.
    """

    def __init__(self, *args, **kwargs) -> None: ...
    def plot(
        self,
        cmap: str | None = None,
        axes: Axes | None = None,
        cbar_axes: Axes | None = None,
        *,
        block: bool = False,
        show: bool | None = None,
        verbose: Incomplete | None = None,
    ):
        """Plot segmentation.

        Parameters
        ----------
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

        Returns
        -------
        fig : Figure
            Matplotlib figure containing the segmentation.
        """

    @property
    def epochs(self) -> BaseEpochs:
        """`~mne.Epochs` instance from which the segmentation was computed."""
