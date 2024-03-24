from pathlib import Path as Path
from typing import Any

from _typeshed import Incomplete
from mne import BaseEpochs as BaseEpochs
from mne.io import BaseRaw as BaseRaw

from .._typing import Picks as Picks
from .._typing import ScalarFloatArray as ScalarFloatArray
from .._typing import ScalarIntArray as ScalarIntArray
from ..utils import _corr_vectors as _corr_vectors
from ..utils._checks import _check_type as _check_type
from ..utils._docs import copy_doc as copy_doc
from ..utils._docs import fill_doc as fill_doc
from ..utils._logs import logger as logger
from ._base import _BaseCluster as _BaseCluster

class AAHCluster(_BaseCluster):
    """Atomize and Agglomerate Hierarchical Clustering (AAHC) algorithm.

    See :footcite:t:`Murray2008` for additional information.

    Parameters
    ----------
    n_clusters : int
        The number of clusters, i.e. the number of microstates.
    normalize_input : bool
        If set, the input data is normalized along the channel dimension.

    References
    ----------
    .. footbibliography::
    """

    _n_clusters: Incomplete
    _cluster_names: Incomplete
    _ignore_polarity: bool
    _normalize_input: Incomplete
    _GEV_: Incomplete

    def __init__(self, n_clusters: int, normalize_input: bool = False) -> None: ...
    def _repr_html_(self, caption: Incomplete | None = None): ...
    def __eq__(self, other: Any) -> bool:
        """Equality == method."""

    def __ne__(self, other: Any) -> bool:
        """Different != method."""

    def _check_fit(self) -> None:
        """Check if the cluster is fitted."""
    _cluster_centers_: Incomplete
    _labels_: Incomplete
    _fitted: bool

    def fit(
        self,
        inst: BaseRaw | BaseEpochs,
        picks: Picks = "eeg",
        tmin: int | float | None = None,
        tmax: int | float | None = None,
        reject_by_annotation: bool = True,
        *,
        verbose: str | None = None,
    ) -> None:
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

    def save(self, fname: str | Path):
        """Save clustering solution to disk.

        Parameters
        ----------
        fname : path-like
            Path to the ``.fif`` file where the clustering solution is saved.
        """

    @staticmethod
    def _aahc(
        data: ScalarFloatArray,
        n_clusters: int,
        ignore_polarity: bool,
        normalize_input: bool,
    ) -> tuple[float, ScalarFloatArray, ScalarIntArray]:
        """Run the AAHC algorithm."""

    @staticmethod
    def _compute_maps(
        data: ScalarFloatArray,
        n_clusters: int,
        ignore_polarity: bool,
        normalize_input: bool,
    ) -> tuple[ScalarFloatArray, ScalarIntArray]:
        """Compute microstates maps."""

    @property
    def normalize_input(self) -> bool:
        """If set, the input data is normalized along the channel dimension.

        :type: `bool`
        """

    @property
    def GEV_(self) -> float:
        """Global Explained Variance.

        :type: `float`
        """

    @_BaseCluster.fitted.setter
    def fitted(self, fitted) -> None:
        """Fitted state.

        :type: `bool`
        """

    @staticmethod
    def _check_ignore_polarity(ignore_polarity: bool) -> bool:
        """Check that ignore_polarity is a boolean."""

    @staticmethod
    def _check_normalize_input(normalize_input: bool) -> bool:
        """Check that normalize_input is a boolean."""
