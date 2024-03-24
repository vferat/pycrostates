from abc import ABC, abstractmethod

from _typeshed import Incomplete
from matplotlib.axes import Axes as Axes
from mne import BaseEpochs
from mne.io import BaseRaw

from .._typing import ScalarFloatArray as ScalarFloatArray
from .._typing import ScalarIntArray as ScalarIntArray
from ..utils import _corr_vectors as _corr_vectors
from ..utils._checks import _check_type as _check_type
from ..utils._docs import fill_doc as fill_doc
from ..utils._logs import logger as logger
from .entropy import entropy as entropy
from .transitions import (
    _compute_expected_transition_matrix as _compute_expected_transition_matrix,
)
from .transitions import _compute_transition_matrix as _compute_transition_matrix

class _BaseSegmentation(ABC):
    """Base class for a Microstates segmentation.

    Parameters
    ----------
    labels : array of shape (n_samples, ) or (n_epochs, n_samples)
        Microstates labels attributed to each sample, i.e. the segmentation.
    inst : Raw | Epochs
        MNE instance used to predict the segmentation.
    cluster_centers : array (n_clusters, n_channels)
         Clusters, i.e. the microstates maps used to compute the segmentation.
    cluster_names : list | None
        Name of the clusters.
    predict_parameters : dict | None
        The prediction parameters.
    """

    _labels: Incomplete
    _inst: Incomplete
    _cluster_centers_: Incomplete
    _cluster_names: Incomplete
    _predict_parameters: Incomplete

    @abstractmethod
    def __init__(
        self,
        labels: ScalarIntArray,
        inst: BaseRaw | BaseEpochs,
        cluster_centers_: ScalarFloatArray,
        cluster_names: list[str] | None = None,
        predict_parameters: dict | None = None,
    ): ...
    def __repr__(self) -> str: ...
    def _repr_html_(self, caption: Incomplete | None = None): ...
    def compute_parameters(self, norm_gfp: bool = True, return_dist: bool = False):
        """Compute microstate parameters.

        .. warning::

            When working with `~mne.Epochs`, this method will put together segments of
            all epochs. This could lead to wrong interpretation especially on state
            durations. To avoid this behaviour, make sure to set the ``reject_edges``
            parameter to ``True`` when creating the segmentation.

        Parameters
        ----------
        norm_gfp : bool
            If True, the :term:`global field power` (GFP) is normalized.
        return_dist : bool
            If True, return the parameters distributions.

        Returns
        -------
        dict : dict
            Dictionaries containing microstate parameters as key/value pairs.
            Keys are named as follow: ``'{microstate name}_{parameter name}'``.

            Available parameters are listed below:

            * ``mean_corr``: Mean correlation value for each time point assigned to a
              given state.
            * ``gev``: Global explained variance expressed by a given state.
              It is the sum of global explained variance values of each time point
              assigned to a given state.
            * ``timecov``: Time coverage, the proportion of time during which
              a given state is active. This metric is expressed as a ratio.
            * ``meandurs``: Mean durations of segments assigned to a given
              state. This metric is expressed in seconds (s).
            * ``occurrences``: Occurrences per second, the mean number of
              segment assigned to a given state per second. This metrics is expressed
              in segment per second.
            * ``dist_corr`` (req. ``return_dist=True``): Distribution of
              correlations values of each time point assigned to a given state.
            * ``dist_gev`` (req. ``return_dist=True``): Distribution of global
              explained variances values of each time point assigned to a given state.
            * ``dist_durs`` (req. ``return_dist=True``): Distribution of
              durations of each segments assigned to a given state. Each value is
              expressed in seconds (s).
        """

    def compute_transition_matrix(
        self, stat: str = "probability", ignore_repetitions: bool = True
    ):
        """Compute the observed transition matrix.

        Count the number of transitions from one state to another and aggregate the
        result as statistic. Transition "from" and "to" unlabeled segments ``-1`` are
        ignored.

        Parameters
        ----------
        %(stat_transition)s
        %(ignore_repetitions)s

        Returns
        -------
        %(transition_matrix)s

        Notes
        -----
        .. warning::

            When working with `~mne.Epochs`, this method will take into account
            transitions that occur between epochs. This could lead to wrong
            interpretation when working with discontinuous data. To avoid this
            behaviour, make sure to set the ``reject_edges`` parameter to ``True`` when
            predicting the segmentation.
        """

    def compute_expected_transition_matrix(
        self, stat: str = "probability", ignore_repetitions: bool = True
    ):
        """Compute the expected transition matrix.

        Compute the theoretical transition matrix as if time course was ignored, but
        microstate proportions kept (i.e. shuffled segmentation). This matrix can be
        used to quantify/correct the effect of microstate time coverage on the observed
        transition matrix obtained with the method
        ``compute_expected_transition_matrix``.
        Transition "from" and "to" unlabeled segments ``-1`` are ignored.

        Parameters
        ----------
        stat : str
            Aggregate statistic to compute transitions. Can be:

            * ``probability`` or ``proportion``: normalize count such as the probabilities along
              the first axis is always equal to ``1``.
            * ``percent``: normalize count such as the probabilities along the first axis is
              always equal to ``100``.
        ignore_repetitions : bool
            If ``True``, ignores state repetitions.
            For example, the input sequence ``AAABBCCD``
            will be transformed into ``ABCD`` before any calculation.
            This is equivalent to setting the duration of all states to 1 sample.

        Returns
        -------
        T : array of shape ``(n_cluster, n_cluster)``
            Array of transition probability values from one label to another.
            First axis indicates state ``"from"``. Second axis indicates state ``"to"``.
        """

    def entropy(self, ignore_repetitions: bool = False, log_base: float | str = 2):
        """Compute the Shannon entropy of the segmentation.

        Compute the Shannon entropy\\ :footcite:p:`shannon1948mathematical`
        of the microstate symbolic sequence.

        Parameters
        ----------
        ignore_repetitions : bool
            If ``True``, ignores state repetitions.
            For example, the input sequence ``AAABBCCD``
            will be transformed into ``ABCD`` before any calculation.
            This is equivalent to setting the duration of all states to 1 sample.
        log_base : float | str
            The log base to use.
            If string:
            * ``bits``: log_base = ``2``
            * ``natural``: log_base = ``np.e``
            * ``dits``: log_base = ``10``
            Default to ``bits``.

        Returns
        -------
        h : float
            The Shannon entropy.

        References
        ----------
        .. footbibliography::
        """

    def plot_cluster_centers(
        self, axes: Axes | None = None, *, block: bool = False, show: bool | None = None
    ):
        """Plot cluster centers as topographic maps.

        Parameters
        ----------
        axes : Axes | None
            Either ``None`` to create a new figure or axes (or an array of axes) on which the
            topographic map should be plotted. If the number of microstates maps to plot is
            ``≥ 1``, an array of axes of size ``n_clusters`` should be provided.
        block : bool
            Whether to halt program execution until the figure is closed.
        show : bool | None
            If True, the figure is shown. If None, the figure is shown if the matplotlib backend
            is interactive.

        Returns
        -------
        fig : Figure
            Matplotlib figure containing the topographic plots.
        """

    @staticmethod
    def _check_cluster_names(
        cluster_names: list[str], cluster_centers_: ScalarFloatArray
    ):
        """Check that the argument 'cluster_names' is valid."""

    @staticmethod
    def _check_predict_parameters(predict_parameters: dict):
        """Check that the argument 'predict_parameters' is valid."""

    @property
    def predict_parameters(self) -> dict:
        """Parameters used to predict the current segmentation.

        :type: `dict`
        """

    @property
    def labels(self) -> ScalarIntArray:
        """Microstate label attributed to each sample (the segmentation).

        :type: `~numpy.array`
        """

    @property
    def cluster_centers_(self) -> ScalarFloatArray:
        """Cluster centers (i.e topographies) used to compute the segmentation.

        :type: `~numpy.array`
        """

    @property
    def cluster_names(self) -> list[str]:
        """Name of the cluster centers.

        :type: `list`
        """
