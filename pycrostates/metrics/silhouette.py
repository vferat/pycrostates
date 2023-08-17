"""Silhouette score."""

import numpy as np
from sklearn.metrics import silhouette_score as sk_silhouette_score

from ..cluster._base import _BaseCluster
from ..utils import _distance_matrix
from ..utils._checks import _check_type
from ..utils._docs import fill_doc


@fill_doc
def silhouette_score(cluster):  # higher the better
    r"""Compute the mean Silhouette Coefficient.

    This function computes the Silhouette Coefficient\ :footcite:p:`Silhouettes` with
    :func:`sklearn.metrics.silhouette_score` from a fitted :ref:`Clustering` instance.

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
    .. footbibliography::
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
