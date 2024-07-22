from __future__ import annotations  # c.f. PEP 563, PEP 649

from itertools import groupby
from typing import TYPE_CHECKING

import numpy as np

from ..utils._checks import _check_type, _check_value
from ..utils._docs import fill_doc

if TYPE_CHECKING:
    from .._typing import ScalarFloatArray, ScalarIntArray


@fill_doc
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
    %(labels_transition)s
    %(n_clusters)s
    %(stat_transition)s
    %(ignore_repetitions)s

    Returns
    -------
    %(transition_matrix)s
    """
    _check_labels_n_clusters(labels, n_clusters)
    return _compute_transition_matrix(
        labels,
        n_clusters,
        stat,
        ignore_repetitions,
    )


def _compute_transition_matrix(
    labels: ScalarIntArray,
    n_clusters: int,
    stat: str = "probability",
    ignore_repetitions: bool = True,
) -> ScalarFloatArray:
    """Compute observed transition."""
    # common error checking
    _check_value(stat, ("count", "probability", "proportion", "percent"), "stat")
    _check_type(ignore_repetitions, (bool,), "ignore_repetitions")

    # reshape if epochs (returns a view)
    labels = labels.reshape(-1)
    # ignore transition to itself (i.e. 1 -> 1)
    if ignore_repetitions:
        labels = [s for s, _ in groupby(labels)]

    T = np.zeros(shape=(n_clusters, n_clusters))
    # number of transitions
    for i, j in zip(labels, labels[1:]):
        if i == -1 or j == -1:
            continue  # ignore unlabeled
        T[i, j] += 1

    # transform to probability
    if stat != "count":
        with np.errstate(divide="ignore", invalid="ignore"):
            T = T / T.sum(axis=1, keepdims=True)
            np.nan_to_num(T, nan=0, posinf=0, copy=False)
        if stat == "percent":
            T = T * 100

    return T


@fill_doc
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
    %(labels_transition)s
    %(n_clusters)s
    %(stat_expected_transitions)s
    %(ignore_repetitions)s

    Returns
    -------
    %(transition_matrix)s
    """
    _check_labels_n_clusters(labels, n_clusters)
    return _compute_expected_transition_matrix(
        labels,
        n_clusters,
        stat,
        ignore_repetitions,
    )


def _compute_expected_transition_matrix(
    labels: ScalarIntArray,
    n_clusters: int,
    stat: str = "probability",
    ignore_repetitions: bool = True,
) -> ScalarFloatArray:
    """Compute theoretical transition matrix.

    The theoretical transition matrix takes into account the time coverage.
    """
    # common error checking
    _check_value(stat, ("probability", "proportion", "percent"), "stat")
    _check_type(ignore_repetitions, (bool,), "ignore_repetitions")

    # reshape if epochs (returns a view)
    labels = labels.reshape(-1)
    states = np.arange(-1, n_clusters)
    # expected probabilities
    T_expected = np.zeros(shape=(states.size, states.size))
    for state_from in states:
        n_from = np.sum(labels == state_from)  # no state_from in labels
        for state_to in states:
            n_to = np.sum(labels == state_to)
            if n_from != 0:
                if state_from != state_to:
                    T_expected[state_from, state_to] = n_to
                else:
                    T_expected[state_from, state_to] = n_to - 1
            else:
                T_expected[state_from, state_to] = 0

    # ignore unlabeled
    T_expected = T_expected[:-1, :-1]
    if ignore_repetitions:
        np.fill_diagonal(T_expected, 0)
    # transform to probability
    with np.errstate(divide="ignore", invalid="ignore"):
        T_expected = T_expected / T_expected.sum(axis=1, keepdims=True)
        np.nan_to_num(T_expected, nan=0, posinf=0, copy=False)
    if stat == "percent":
        T_expected = T_expected * 100

    return T_expected


def _check_labels_n_clusters(
    labels: ScalarIntArray,
    n_clusters: int,
) -> None:
    """Checker for labels and n_clusters."""
    _check_type(labels, (np.ndarray,), "labels")
    _check_type(n_clusters, ("int",), "n_clusters")
    if n_clusters <= 0:
        raise ValueError(
            "The provided number of clusters 'n_clusters' must be a strictly "
            f"positive integer. '{n_clusters}' is invalid."
        )

    # only accept array of integers to simplify maintenance
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(
            "The argument 'labels' must contain the labels of each timepoint "
            "encoded as consecutive positive integers (0-indexed). Make sure "
            f"you are providing an integer array. '{labels.dtype}' is invalid."
        )
    # check for negative integers except -1
    if np.any(labels < -1):
        raise ValueError(
            "The argument 'labels' must contain the labels of each timepoint "
            "encoded as consecutive positive integers (0-indexed) with '-1' "
            "representing unlabelled segments. Negative integers except -1 "
            "are invalid."
        )
    # check that the labels are in the range of clusters
    states = sorted(elt for elt in np.unique(labels) if 0 <= elt)
    isin = np.isin(states, np.arange(n_clusters))
    if not np.all(isin):
        states = np.array(states)[~isin]
        raise ValueError(
            "The argument 'labels' must contain the labels of each timepoint "
            "encoded as consecutive positive integers (0-indexed), between 0 "
            f"and 'n_clusters - 1' ({n_clusters - 1}). "
            f"'{states}' {'is' if len(states) == 1 else 'are'} invalid."
        )
