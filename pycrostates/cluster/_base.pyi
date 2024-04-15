from abc import ABC, abstractmethod
from pathlib import Path as Path
from typing import Any

import numpy as np
from _typeshed import Incomplete
from matplotlib.axes import Axes as Axes
from mne import BaseEpochs
from mne.io import BaseRaw
from numpy.typing import NDArray as NDArray

from .._typing import AxesArray as AxesArray
from .._typing import Picks as Picks
from .._typing import ScalarFloatArray as ScalarFloatArray
from .._typing import ScalarIntArray as ScalarIntArray
from ..io import ChData as ChData
from ..segmentation import EpochsSegmentation as EpochsSegmentation
from ..segmentation import RawSegmentation as RawSegmentation
from ..utils import _corr_vectors as _corr_vectors
from ..utils._checks import _check_picks_uniqueness as _check_picks_uniqueness
from ..utils._checks import _check_reject_by_annotation as _check_reject_by_annotation
from ..utils._checks import _check_tmin_tmax as _check_tmin_tmax
from ..utils._checks import _check_type as _check_type
from ..utils._checks import _check_value as _check_value
from ..utils._docs import fill_doc as fill_doc
from ..utils._logs import logger as logger
from ..utils.mixin import ChannelsMixin as ChannelsMixin
from ..utils.mixin import ContainsMixin as ContainsMixin
from ..utils.mixin import MontageMixin as MontageMixin
from .utils import optimize_order as optimize_order

class _BaseCluster(ABC, ChannelsMixin, ContainsMixin, MontageMixin):
    """Base Class for Microstates Clustering algorithms."""

    _n_clusters: Incomplete
    _cluster_names: Incomplete
    _cluster_centers_: Incomplete
    _ignore_polarity: Incomplete
    _info: Incomplete
    _fitted_data: Incomplete
    _labels_: Incomplete
    _fitted: bool

    @abstractmethod
    def __init__(self): ...
    def __repr__(self) -> str:
        """String representation."""

    def _repr_html_(self, caption: Incomplete | None = None):
        """HTML representation."""

    def __eq__(self, other: Any) -> bool:
        """Equality == method."""

    def __ne__(self, other: Any) -> bool:
        """Different != method."""

    def copy(self, deep: bool = True):
        """Return a copy of the instance.

        Parameters
        ----------
        deep : bool
            If True, `~copy.deepcopy` is used instead of `~copy.copy`.
        """

    def _check_fit(self) -> None:
        """Check if the cluster is fitted."""

    def _check_unfitted(self) -> None:
        """Check if the cluster is unfitted."""

    @abstractmethod
    def fit(
        self,
        inst: BaseRaw | BaseEpochs | ChData,
        picks: Picks = "eeg",
        tmin: int | float | None = None,
        tmax: int | float | None = None,
        reject_by_annotation: bool = True,
        *,
        verbose: str | None = None,
    ) -> ScalarFloatArray:
        """Compute cluster centers.

        Parameters
        ----------
        inst : Raw | Epochs | ChData
            MNE `~mne.io.Raw`, `~mne.Epochs` or `~pycrostates.io.ChData` object
            from which to extract :term:`cluster centers`.
        picks : str | list | slice | None
            Channels to include. Note that all channels selected must have the same
            type. Slices and lists of integers will be interpreted as channel indices.
            In lists, channel name strings (e.g. ``['Fp1', 'Fp2']``) will pick the given
            channels. Can also be the string values ``“all”`` to pick all channels, or
            ``“data”`` to pick data channels. ``"eeg"`` (default) will pick all eeg
            channels. Note that channels in ``info['bads']`` will be included if their
            names or indices are explicitly provided.
        tmin : float
            Start time of the raw data to use in seconds (must be >= 0).
        tmax : float
            End time of the raw data to use in seconds (cannot exceed data duration).
        reject_by_annotation : bool
            Whether to omit bad segments from the data before fitting. If ``True``
            (default), annotated segments whose description begins with ``'bad'`` are
            omitted. If ``False``, no rejection based on annotations is performed.

            Has no effect if ``inst`` is not a :class:`mne.io.Raw` object.
        verbose : int | str | bool | None
            Sets the verbosity level. The verbosity increases gradually between ``"CRITICAL"``,
            ``"ERROR"``, ``"WARNING"``, ``"INFO"`` and ``"DEBUG"``. If None is provided, the
            verbosity is set to ``"WARNING"``. If a bool is provided, the verbosity is set to
            ``"WARNING"`` for False and to ``"INFO"`` for True.
        """

    def rename_clusters(
        self,
        mapping: dict[str, str] | None = None,
        new_names: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        """Rename the clusters.

        Parameters
        ----------
        mapping : dict
            Mapping from the old names to the new names. The keys are the old names and
            the values are the new names.
        new_names : list | tuple
            1D iterable containing the new cluster names. The length of the iterable
            should match the number of clusters.

        Notes
        -----
        Operates in-place.
        """

    def reorder_clusters(
        self,
        mapping: dict[int, int] | None = None,
        order: list[int] | tuple[int, ...] | ScalarIntArray | None = None,
        template: _BaseCluster | None = None,
    ) -> None:
        """
        Reorder the clusters of the fitted model.

        Specify one of the following arguments to change the current order:

        * ``mapping``: a dictionary that maps old cluster positions to new positions,
        * ``order``: a 1D iterable containing the new order,
        * ``template``: a fitted clustering algorithm used as a reference to match the
          order.

        Only one argument can be set at a time.

        Parameters
        ----------
        mapping : dict
            Mapping from the old order to the new order.
            key: old position, value: new position.
        order : list of int | tuple of int | array of int
            1D iterable containing the new order. Positions are 0-indexed.
        template : :ref:`cluster`
            Fitted clustering algorithm use as template for ordering optimization. For
            more details about the current implementation, check the
            :func:`pycrostates.cluster.utils.optimize_order` documentation.

        Notes
        -----
        Operates in-place.
        """

    def invert_polarity(
        self, invert: bool | list[bool] | tuple[bool, ...] | NDArray[np.bool]
    ) -> None:
        """Invert map polarities.

        Parameters
        ----------
        invert : bool | list of bool | array of bool
            List of bool of length ``n_clusters``.
            True will invert map polarity, while False will have no effect.
            If a `bool` is provided, it is applied to all maps.

        Notes
        -----
        Operates in-place.

        Inverting polarities has no effect on the other steps of the analysis as
        polarity is ignored in the current methodology. This function is only used for
        tuning visualization (i.e. for visual inspection and/or to generate figure for
        an article).
        """

    def plot(
        self,
        axes: Axes | AxesArray | None = None,
        show_gradient: bool | None = False,
        gradient_kwargs: dict[str, Any] = {
            "color": "black",
            "linestyle": "-",
            "marker": "P",
        },
        *,
        block: bool = False,
        show: bool | None = None,
        verbose: str | None = None,
        **kwargs,
    ):
        """
        Plot cluster centers as topographic maps.

        Parameters
        ----------
        axes : Axes | None
            Either ``None`` to create a new figure or axes (or an array of axes) on which the
            topographic map should be plotted. If the number of microstates maps to plot is
            ``≥ 1``, an array of axes of size ``n_clusters`` should be provided.
        show_gradient : bool
            If True, plot a line between channel locations with highest and lowest
            values.
        gradient_kwargs : dict
            Additional keyword arguments passed to :meth:`matplotlib.axes.Axes.plot` to
            plot gradient line.
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
            Additional keyword arguments are passed to :func:`mne.viz.plot_topomap`.

        Returns
        -------
        f : Figure
            Matplotlib figure containing the topographic plots.
        """

    @abstractmethod
    def save(self, fname: str | Path):
        """Save clustering solution to disk.

        Parameters
        ----------
        fname : path-like
            Path to the ``.fif`` file where the clustering solution is saved.
        """

    def predict(
        self,
        inst: BaseRaw | BaseEpochs,
        picks: Picks = None,
        factor: int = 0,
        half_window_size: int = 1,
        tol: int | float = 1e-05,
        min_segment_length: int = 0,
        reject_edges: bool = True,
        reject_by_annotation: bool = True,
        *,
        verbose: str | None = None,
    ):
        """Segment `~mne.io.Raw` or `~mne.Epochs` into microstate sequence.

        Segment instance into microstate sequence using the segmentation smoothing
        algorithm\\ :footcite:p:`Marqui1995`.

        Parameters
        ----------
        inst : Raw | Epochs
            MNE `~mne.io.Raw` or `~mne.Epochs` object containing the data to use for
            prediction.
        picks : str | list | slice | None
            Channels to include. Note that all channels selected must have the same
            type. Slices and lists of integers will be interpreted as channel indices.
            In lists, channel name strings (e.g. ``['Fp1', 'Fp2']``) will pick the given
            channels. Can also be the string values ``“all”`` to pick all channels, or
            ``“data”`` to pick data channels. ``None`` (default) will pick all channels
            used during fitting (e.g., ``self.info['ch_names']``). Note that channels in
            ``info['bads']`` will be included if their names or indices are explicitly
            provided.
        factor : int
            Factor used for label smoothing. ``0`` means no smoothing. Default to 0.
        half_window_size : int
            Number of samples used for the half window size while smoothing labels. The
            half window size is defined as ``window_size = 2 * half_window_size + 1``.
            It has no effect if ``factor=0`` (default). Default to 1.
        tol : float
            Convergence tolerance.
        min_segment_length : int
            Minimum segment length (in samples). If a segment is shorter than this
            value, it will be recursively reasigned to neighbouring segments based on
            absolute spatial correlation.
        reject_edges : bool
            If ``True``, set first and last segments to unlabeled.
        reject_by_annotation : bool
            Whether to omit bad segments from the data before fitting. If ``True``
            (default), annotated segments whose description begins with ``'bad'`` are
            omitted. If ``False``, no rejection based on annotations is performed.

            Has no effect if ``inst`` is not a :class:`mne.io.Raw` object.
        verbose : int | str | bool | None
            Sets the verbosity level. The verbosity increases gradually between ``"CRITICAL"``,
            ``"ERROR"``, ``"WARNING"``, ``"INFO"`` and ``"DEBUG"``. If None is provided, the
            verbosity is set to ``"WARNING"``. If a bool is provided, the verbosity is set to
            ``"WARNING"`` for False and to ``"INFO"`` for True.

        Returns
        -------
        segmentation : RawSegmentation | EpochsSegmentation
            Microstate sequence derivated from instance data. Timepoints are labeled
            according to cluster centers number: ``0`` for the first center, ``1`` for
            the second, etc.. ``-1`` is used for unlabeled time points.

        References
        ----------
        .. footbibliography::
        """

    def _predict_raw(
        self,
        raw: BaseRaw,
        picks_data: ScalarIntArray,
        factor: int,
        tol: int | float,
        half_window_size: int,
        min_segment_length: int,
        reject_edges: bool,
        reject_by_annotation: bool,
    ) -> RawSegmentation:
        """Create segmentation for raw."""

    def _predict_epochs(
        self,
        epochs: BaseEpochs,
        picks_data: ScalarIntArray,
        factor: int,
        tol: int | float,
        half_window_size: int,
        min_segment_length: int,
        reject_edges: bool,
    ) -> EpochsSegmentation:
        """Create segmentation for epochs."""

    @staticmethod
    def _segment(
        data: ScalarFloatArray,
        states: ScalarFloatArray,
        factor: int,
        tol: int | float,
        half_window_size: int,
    ) -> ScalarIntArray:
        """Create segmentation. Must operate on a copy of states."""

    @staticmethod
    def _smooth_segmentation(
        data: ScalarFloatArray,
        states: ScalarFloatArray,
        labels: ScalarIntArray,
        factor: int,
        tol: int | float,
        half_window_size: int,
    ) -> ScalarIntArray:
        """Apply smoothing.

        Adapted from [1].

        References
        ----------
        .. [1] R. D. Pascual-Marqui, C. M. Michel and D. Lehmann.
               Segmentation of brain electrical activity into microstates:
               model estimation and validation.
               IEEE Transactions on Biomedical Engineering,
               vol. 42, no. 7, pp. 658-665, July 1995,
               https://doi.org/10.1109/10.391164.
        """

    @staticmethod
    def _reject_short_segments(
        segmentation: ScalarIntArray, data: ScalarFloatArray, min_segment_length: int
    ) -> ScalarIntArray:
        """Reject segments that are too short.

        Reject segments that are too short by replacing the labels with the adjacent
        labels based on data correlation.
        """

    @staticmethod
    def _reject_edge_segments(segmentation: ScalarIntArray) -> ScalarIntArray:
        """Set the first and last segment as unlabeled (0)."""

    @property
    def n_clusters(self) -> int:
        """Number of clusters (number of microstates).

        :type: `int`
        """

    @property
    def info(self):
        """Info instance with the channel information used to fit the instance.

        :type: `~pycrostates.io.ChInfo`
        """

    @property
    def fitted(self) -> bool:
        """Fitted state.

        :type: `bool`
        """

    @fitted.setter
    def fitted(self, fitted) -> None:
        """Fitted state.

        :type: `bool`
        """

    @property
    def cluster_centers_(self) -> ScalarFloatArray:
        """Fitted clusters (the microstates maps).

        Returns None if cluster algorithm has not been fitted.

        :type: `~numpy.array` of shape (n_clusters, n_channels) | None
        """

    @property
    def fitted_data(self) -> ScalarFloatArray:
        """Data array used to fit the clustering algorithm.

        :type: `~numpy.array` of shape (n_channels, n_samples) | None
        """

    @property
    def labels_(self) -> ScalarIntArray:
        """Microstate label attributed to each sample of the fitted data.

        :type: `~numpy.array` of shape (n_samples, ) | None
        """

    @property
    def cluster_names(self) -> list[str]:
        """Name of the clusters.

        :type: `list`
        """

    @cluster_names.setter
    def cluster_names(self, other: Any):
        """Name of the clusters.

        :type: `list`
        """

    @staticmethod
    def _check_n_clusters(n_clusters: int) -> int:
        """Check that the number of clusters is a positive integer."""
