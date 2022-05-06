"""Metric functions for evaluating clusters."""

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder


def _distance_matrix(X, Y=None):
    distances = np.abs(1 / np.corrcoef(X, Y)) - 1
    distances = np.nan_to_num(
        distances, copy=False, nan=10e300, posinf=10e300, neginf=-10e300
    )
    # TODO: Sure about the 10e300? That's 1e301.
    return distances


def silhouette(modK):  # lower the better
    """
    Compute the mean Silhouette Coefficient of a fitted clustering algorithm.

    This function is a proxy function for
    :func:`sklearn.metrics.silhouette_score` that applies directly to a fitted
    :class:´pycrostate.clustering.BaseClustering´. It uses the absolute spatial
    correlation for distance computations.

    Parameters
    ----------
    BaseClustering : :class:`pycrostate.clustering.BaseClustering`
        Fitted clustering algorithm from which to compute score.

    Returns
    -------
    silhouette : float
        Mean Silhouette Coefficient.

    Notes
    -----
    For more details regarding the implementation, please refere to
    :func:`sklearn.metrics.silhouette_score`.
    This proxy function uses metric="precomputed" with the absolute spatial
    correlation for distance computations.

    References
    ----------
    .. [1] `Peter J. Rousseeuw (1987).
       "Silhouettes: a Graphical Aid to the Interpretation and Validation of
       Cluster Analysis".
       Computational and Applied Mathematics 20: 53-65.
       <https://doi.org/10.1016/0377-0427(87)90125-7>`_
    """
    modK._check_fit()
    data = modK._fitted_data
    labels = modK._labels_
    keep = np.linalg.norm(data.T, axis=1) != 0
    data = data[:, keep]
    labels = labels[keep]
    distances = _distance_matrix(data.T)
    silhouette = silhouette_score(distances, labels, metric="precomputed")
    return silhouette
