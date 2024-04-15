from .._typing import ScalarFloatArray as ScalarFloatArray
from .._typing import ScalarIntArray as ScalarIntArray
from ..utils._checks import _check_type as _check_type
from ..utils._checks import _check_value as _check_value
from ..utils._docs import fill_doc as fill_doc

def compute_transition_matrix(
    labels: ScalarIntArray,
    n_clusters: int,
    stat: str = "probability",
    ignore_repetitions: bool = True,
) -> ScalarFloatArray:
    """Compute the observed transition matrix.

    Count the number of transitions from one state to another and aggregate the result
    as statistic. Transitions "from" and "to" unlabeled segments ``-1`` are ignored.

    Parameters
    ----------
    labels : array of shape ``(n_samples,)`` or ``(n_epochs, n_samples)``
        Microstates labels attributed to each sample, i.e. the segmentation.
    n_clusters : int
        The number of clusters, i.e. the number of microstates.
    stat : str
        Aggregate statistic to compute transitions. Can be:

        * ``count``: show the number of observations of each transition.
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

def _compute_transition_matrix(
    labels: ScalarIntArray,
    n_clusters: int,
    stat: str = "probability",
    ignore_repetitions: bool = True,
) -> ScalarFloatArray:
    """Compute observed transition."""

def compute_expected_transition_matrix(
    labels: ScalarIntArray,
    n_clusters: int,
    stat: str = "probability",
    ignore_repetitions: bool = True,
) -> ScalarFloatArray:
    """Compute the expected transition matrix.

    Compute the theoretical transition matrix as if time course was ignored, but
    microstate proportions was kept (i.e. shuffled segmentation). This matrix can be
    used to quantify/correct the effect of microstate time coverage on the observed
    transition matrix obtained with the
    :func:`pycrostates.segmentation.compute_transition_matrix`.
    Transition "from" and "to" unlabeled segments ``-1`` are ignored.

    Parameters
    ----------
    labels : array of shape ``(n_samples,)`` or ``(n_epochs, n_samples)``
        Microstates labels attributed to each sample, i.e. the segmentation.
    n_clusters : int
        The number of clusters, i.e. the number of microstates.
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

def _compute_expected_transition_matrix(
    labels: ScalarIntArray,
    n_clusters: int,
    stat: str = "probability",
    ignore_repetitions: bool = True,
) -> ScalarFloatArray:
    """Compute theoretical transition matrix.

    The theoretical transition matrix takes into account the time coverage.
    """

def _check_labels_n_clusters(labels: ScalarIntArray, n_clusters: int) -> None:
    """Checker for labels and n_clusters."""
