"""Silhouette score."""

import numpy as np
from sklearn.metrics import silhouette_score as sk_silhouette_score

from ..cluster._base import _BaseCluster
from ..utils import _distance_matrix
from ..utils._checks import _check_type
from ..utils._docs import fill_doc


@fill_doc
def silhouette_score(cluster):  # higher the better
    """Compute the mean Silhouette Coefficient.

    This function is a wrapper around :func:`sklearn.metrics.silhouette_score`
    that applies directly to a fitted :ref:`Clustering` instance. It uses the
    absolute spatial correlation for distance computations.

    Parameters
    ----------
    %(cluster)s

    Returns
    -------
    silhouette : float
        The mean Silhouette Coefficient.

    Notes
    -----
    For more details regarding the implementation, please refer to
    :func:`sklearn.metrics.silhouette_score`.
    This proxy function uses ``metric="precomputed"`` with the absolute spatial
    correlation for distance computations.

    References
    ----------
    .. [1] `Peter J. Rousseeuw (1987).
       "Silhouettes: a Graphical Aid to the Interpretation and Validation of
       Cluster Analysis".
       Computational and Applied Mathematics 20: 53-65.
       <https://doi.org/10.1016/0377-0427(87)90125-7>`_
    """
    _check_type(cluster, (_BaseCluster,), item_name="cluster")
    cluster._check_fit()
    data = cluster._fitted_data
    labels = cluster._labels_
    keep = np.linalg.norm(data.T, axis=1) != 0
    data = data[:, keep]
    labels = labels[keep]
    distances = _distance_matrix(data.T)
    silhouette = sk_silhouette_score(distances, labels, metric="precomputed")
    return silhouette
