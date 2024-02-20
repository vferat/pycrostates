"""Davies Bouldin score."""

import numpy as np

from ..cluster._base import _BaseCluster
from ..utils import _distance
from ..utils._checks import _check_type
from ..utils._docs import fill_doc


@fill_doc
def davies_bouldin_score(cluster):  # lower the better
    r"""Compute the Davies-Bouldin score.

    This function computes the Davies-Bouldin score\ :footcite:p:`Davies-Bouldin` with
    :func:`sklearn.metrics.davies_bouldin_score` from a fitted :ref:`Clustering`
    instance.

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
    :func:`sklearn.metrics.davies_bouldin_score`. This function was modified in order to
    use the absolute spatial correlation for distance computations instead of the
    euclidean distance.

    References
    ----------
    .. footbibliography::
    """
    _check_type(cluster, (_BaseCluster,), item_name="cluster")
    cluster._check_fit()
    data = cluster._fitted_data
    labels = cluster._labels_
    ignore_polarity = cluster._ignore_polarity

    keep = np.linalg.norm(data.T, axis=1) != 0
    data = data[:, keep]
    labels = labels[keep]
    if ignore_polarity:
        x = cluster.cluster_centers_[labels].T
        sign = np.sign((x.T * data.T).sum(axis=1))
        data = data * sign
    davies_bouldin_score = _davies_bouldin_score(
        data.T, labels, ignore_polarity=ignore_polarity
    )
    return davies_bouldin_score


def _davies_bouldin_score(X, labels, ignore_polarity):
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
    # Calculate the number of clusters
    num_clusters = len(set(labels))

    # Calculate the centroids of the clusters
    centroids = np.array([np.mean(X[labels == i], axis=0) for i in range(num_clusters)])

    # Calculate pairwise distances between centroids
    centroid_distances = _distance(centroids, ignore_polarity=ignore_polarity)

    # Initialize array to hold scatter values for each cluster
    scatter_values = np.zeros(num_clusters)

    # Calculate scatter for each cluster
    for i in range(num_clusters):
        cluster_points = X[labels == i]
        cluster_centroid = centroids[i]
        scatter_values[i] = np.mean(
            [_distance(point.reshape(-1, 1), cluster_centroid.reshape(-1, 1)) for point in cluster_points]
        )

    # Initialize array to hold ratio values for each cluster pair
    ratio_values = np.zeros((num_clusters, num_clusters))

    # Calculate ratio for each cluster pair
    for i in range(num_clusters):
        for j in range(num_clusters):
            if i != j:
                ratio_values[i][j] = (
                    scatter_values[i] + scatter_values[j]
                ) / centroid_distances[i][j]

    # Compute Davies-Bouldin Index
    db_index = np.mean(np.max(ratio_values, axis=1))

    return db_index
