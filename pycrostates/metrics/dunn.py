"""Dunn score."""

import numpy as np

from ..cluster._base import _BaseCluster
from ..utils import _distance_matrix
from ..utils._checks import _check_type
from ..utils._docs import fill_doc


@fill_doc
def dunn_score(cluster):  # higher the better
    """Compute the Dunn index score.

    Parameters
    ----------
    %(cluster)s

    Returns
    -------
    score : float
        The resulting Dunn score.

    Notes
    -----
    This function uses the absolute spatial correlation for distance.

    References
    ----------
    .. [1] `J. C. Dunn (1974).
       "Well-Separated Clusters and Optimal Fuzzy Partitions".
       Journal of Cybernetics.
       <https://doi.org/10.1080/01969727408546059>`_
    """
    _check_type(cluster, (_BaseCluster,), item_name="cluster")
    cluster._check_fit()
    data = cluster._fitted_data
    labels = cluster._labels_
    keep = np.linalg.norm(data.T, axis=1) != 0
    data = data[:, keep]
    labels = labels[keep]
    score = _dunn_score(data.T, labels)
    return score


def _dunn_score(X, labels):  # higher the better
    """Compute the Dunn index.

    Parameters
    ----------
    X : np.array
        np.array([N, p]) of all points
    labels: np.array
        np.array([N]) labels of all points

    Notes
    -----
    Based on https://github.com/jqmviegas/jqm_cvi
    """
    distances = _distance_matrix(X)
    ks = np.sort(np.unique(labels))

    deltas = np.ones([len(ks), len(ks)]) * 1000000
    big_deltas = np.zeros([len(ks), 1])

    for i, ks_i in enumerate(ks):
        for j, ks_j in enumerate(ks):
            if i == j:
                continue  # skip diagonal
            deltas[i, j] = _delta_fast(
                (labels == ks_i), (labels == ks_j), distances
            )
        big_deltas[i] = _big_delta_fast((labels == ks_i), distances)

    di = np.min(deltas) / np.max(big_deltas)
    return di


def _delta_fast(ck, cl, distances):
    values = distances[np.where(ck)][:, np.where(cl)]
    values = values[np.nonzero(values)]
    return np.min(values)


def _big_delta_fast(ci, distances):
    values = distances[np.where(ci)][:, np.where(ci)]
    return np.max(values)
