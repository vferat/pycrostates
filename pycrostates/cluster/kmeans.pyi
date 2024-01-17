from pathlib import Path as Path
from typing import Any, Optional, Union

from _typeshed import Incomplete
from mne import BaseEpochs as BaseEpochs
from mne.io import BaseRaw as BaseRaw
from numpy.random import Generator as Generator
from numpy.random import RandomState as RandomState
from numpy.typing import NDArray

from .._typing import CHData as CHData
from .._typing import Picks as Picks
from .._typing import RANDomState as RANDomState
from ..utils import _corr_vectors as _corr_vectors
from ..utils._checks import _check_n_jobs as _check_n_jobs
from ..utils._checks import _check_random_state as _check_random_state
from ..utils._checks import _check_type as _check_type
from ..utils._docs import copy_doc as copy_doc
from ..utils._docs import fill_doc as fill_doc
from ..utils._logs import logger as logger
from ._base import _BaseCluster as _BaseCluster

class ModKMeans(_BaseCluster):
    """Modified K-Means clustering algorithm.

    See :footcite:t:`Marqui1995` for additional information.

    Parameters
    ----------
    n_clusters : int
        The number of clusters, i.e. the number of microstates.
    n_init : int
        Number of time the k-means algorithm is run with different centroid seeds. The
        final result will be the run with the highest Global Explained Variance (GEV).
    max_iter : int
        Maximum number of iterations of the K-means algorithm for a single run.
    tol : float
        Relative tolerance with regards estimate residual noise in the cluster centers
        of two consecutive iterations to declare convergence.
    random_state : None | int | instance of ~numpy.random.RandomState
        A seed for the NumPy random number generator (RNG). If ``None`` (default),
        the seed will be  obtained from the operating system
        (see  :class:`~numpy.random.RandomState` for details), meaning it will most
        likely produce different output every time this function or method is run.
        To achieve reproducible results, pass a value here to explicitly initialize
        the RNG with a defined state.

    References
    ----------
    .. footbibliography::
    """

    _n_clusters: Incomplete
    _cluster_names: Incomplete
    _n_init: Incomplete
    _max_iter: Incomplete
    _tol: Incomplete
    _random_state: Incomplete
    _GEV_: Incomplete

    def __init__(
        self,
        n_clusters: int,
        n_init: int = 100,
        max_iter: int = 300,
        tol: Union[int, float] = 1e-06,
        random_state: RANDomState = None,
    ) -> None: ...
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
    _ignore_polarity: bool

    def fit(
        self,
        inst: Union[BaseRaw, BaseEpochs, CHData],
        picks: Picks = "eeg",
        tmin: Optional[Union[int, float]] = None,
        tmax: Optional[Union[int, float]] = None,
        reject_by_annotation: bool = True,
        n_jobs: int = 1,
        *,
        verbose: Optional[str] = None,
    ) -> None:
        """Compute cluster centers.

        Parameters
        ----------
        inst : Raw | Epochs | ChData
            MNE `~mne.io.Raw`, `~mne.Epochs` or `~pycrostates.io.ChData` object from
            which to extract :term:`cluster centers`.
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
        n_jobs : int | None
            The number of jobs to run in parallel. If ``-1``, it is set
            to the number of CPU cores. Requires the :mod:`joblib` package.
            ``None`` (default) is a marker for 'unset' that will be interpreted
            as ``n_jobs=1`` (sequential execution) unless the call is performed under
            a :class:`joblib:joblib.parallel_config` context manager that sets another
            value for ``n_jobs``.
        verbose : int | str | bool | None
            Sets the verbosity level. The verbosity increases gradually between ``"CRITICAL"``,
            ``"ERROR"``, ``"WARNING"``, ``"INFO"`` and ``"DEBUG"``. If None is provided, the
            verbosity is set to ``"WARNING"``. If a bool is provided, the verbosity is set to
            ``"WARNING"`` for False and to ``"INFO"`` for True.
        """

    def save(self, fname: Union[str, Path]):
        """Save clustering solution to disk.

        Parameters
        ----------
        fname : path-like
            Path to the ``.fif`` file where the clustering solution is saved.
        """

    @staticmethod
    def _kmeans(
        data: NDArray[float],
        n_clusters: int,
        max_iter: int,
        random_state: Union[RandomState, Generator],
        tol: Union[int, float],
    ) -> tuple[float, NDArray[float], NDArray[int], bool]:
        """Run the k-means algorithm."""

    @staticmethod
    def _compute_maps(
        data: NDArray[float],
        n_clusters: int,
        max_iter: int,
        random_state: Union[RandomState, Generator],
        tol: Union[int, float],
    ) -> tuple[NDArray[float], bool]:
        """Compute microstates maps.

        Based on mne_microstates by Marijn van Vliet <w.m.vanvliet@gmail.com>
        https://github.com/wmvanvliet/mne_microstates/blob/master/microstates.py
        """

    @property
    def n_init(self) -> int:
        """Number of k-means algorithms run with different centroid seeds.

        :type: `int`
        """

    @property
    def max_iter(self) -> int:
        """Maximum number of iterations of the k-means algorithm for a run.

        :type: `int`
        """

    @property
    def tol(self) -> Union[int, float]:
        """Relative tolerance to reach convergence.

        :type: `float`
        """

    @property
    def random_state(self) -> Union[RandomState, Generator]:
        """Random state to fix seed generation.

        :type: `~numpy.random.RandomState` | `~numpy.random.Generator`
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
    def _check_n_init(n_init: int) -> int:
        """Check that n_init is a positive integer."""

    @staticmethod
    def _check_max_iter(max_iter: int) -> int:
        """Check that max_iter is a positive integer."""

    @staticmethod
    def _check_tol(tol: Union[int, float]) -> Union[int, float]:
        """Check that tol is a positive number."""
