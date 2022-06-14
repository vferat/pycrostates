"""Davies Bouldin score."""

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import _safe_indexing, check_X_y

from ..cluster._base import _BaseCluster
from ..utils import _distance_matrix
from ..utils._checks import _check_type
from ..utils._docs import fill_doc


@fill_doc
def davies_bouldin_score(cluster):  # lower the better
    """Compute the Davies-Bouldin score.

    This function is a wrapper around
    :func:`sklearn.metrics.davies_bouldin_score` that applies directly to a
    fitted :ref:`Clustering` instance. It uses the absolute spatial correlation
    for distance computations.

    Parameters
    ----------
    %(cluster)s

    Returns
    -------
    score : float
        The resulting Davies-Bouldin score.

    Notes
    -----
    For more details regarding the implementation, please refer to
    :func:`sklearn.metrics.davies_bouldin_score`.
    This function was modified in order to use the absolute spatial correlation
    for distance computations instead of the euclidean distance.

    References
    ----------
    .. [1] `Davies, David L.; Bouldin, Donald W (1979).
       "A Cluster Separation Measure"
       IEEE Transactions on Pattern Analysis and Machine Intelligence.
       PAMI-1 (2): 224-227.
       <https://doi.org/10.1109/TPAMI.1979.4766909>`_
    """
    _check_type(cluster, (_BaseCluster,), item_name="cluster")
    cluster._check_fit()
    data = cluster._fitted_data
    labels = cluster._labels_
    keep = np.linalg.norm(data.T, axis=1) != 0
    data = data[:, keep]
    labels = labels[keep]
    # Align polarities just in case..
    x = cluster.cluster_centers_[labels].T
    sign = np.sign((x.T * data.T).sum(axis=1))
    data = data * sign
    davies_bouldin_score = _davies_bouldin_score(data.T, labels)
    return davies_bouldin_score


def _davies_bouldin_score(X, labels):
    """Compute the Davies-Bouldin score.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        A list of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.
    labels : array of shape (n_samples,)
        Predicted labels for each sample.

    Returns
    -------
    score: float
        The resulting Davies-Bouldin score.
    """
    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_labels = len(le.classes_)

    intra_dists = np.zeros(n_labels)
    centroids = np.zeros((n_labels, len(X[0])), dtype=float)
    for k in range(n_labels):
        cluster_k = _safe_indexing(X, labels == k)
        centroid = cluster_k.mean(axis=0)
        centroids[k] = centroid
        intra_dists[k] = np.average(_distance_matrix(cluster_k, [centroid]))

    centroid_distances = _distance_matrix(centroids)

    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0.0

    centroid_distances[centroid_distances == 0] = np.inf
    combined_intra_dists = intra_dists[:, None] + intra_dists
    scores = np.max(combined_intra_dists / centroid_distances, axis=1)
    return np.mean(scores)
