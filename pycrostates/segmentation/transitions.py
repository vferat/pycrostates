from itertools import groupby

import numpy as np
from numpy.typing import NDArray

from ..utils._checks import _check_type, _check_value


def compute_transition_matrix(
    labels: NDArray[int],
    n_clusters: int,
    stat: str = " probability",
    ignore_self: bool = True,
) -> NDArray[float]:
    # TODO: Error checking on labels/n_clusters
    return _compute_transition_matrix(
        labels,
        n_clusters,
        stat,
        ignore_self,
    )


def _compute_transition_matrix(
    labels: NDArray[int],
    n_clusters: int,
    stat: str = " probability",
    ignore_self: bool = True,
) -> NDArray[float]:
    """Compute observed transition."""
    # common error checking
    _check_value(
        stat, ("count", "probability", "proportion", "percent"), "stat"
    )
    _check_type(ignore_self, (bool,), "ignore_self")

    # reshape if epochs (returns a view)
    labels = labels.reshape(-1)
    # ignore transition to itself (i.e. 1 -> 1)
    if ignore_self:
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


def compute_expected_transition_matrix(
    labels: NDArray[int],
    n_clusters: int,
    stat: str = " probability",
    ignore_self: bool = True,
) -> NDArray[float]:
    # TODO: Error checking on labels/n_clusters
    return _compute_expected_transition_matrix(
        labels,
        n_clusters,
        stat,
        ignore_self,
    )


def _compute_expected_transition_matrix(
    labels: NDArray[int],
    n_clusters: int,
    stat: str = " probability",
    ignore_self: bool = True,
) -> NDArray[float]:
    """Compute theoretical transition matrix.

    The theoretical transition matrix takes into account the time coverage.
    """
    # common error checking
    _check_value(stat, ("probability", "proportion", "percent"), "stat")
    _check_type(ignore_self, (bool,), "ignore_self")

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
    if ignore_self:
        np.fill_diagonal(T_expected, 0)
    # transform to probability
    with np.errstate(divide="ignore", invalid="ignore"):
        T_expected = T_expected / T_expected.sum(axis=1, keepdims=True)
        np.nan_to_num(T_expected, nan=0, posinf=0, copy=False)
    if stat == "percent":
        T_expected = T_expected * 100

    return T_expected
