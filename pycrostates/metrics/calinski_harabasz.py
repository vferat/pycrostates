"""Calinski Harabasz score."""

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y

from ..cluster._base import _BaseCluster
from ..utils import _distance_matrix
from ..utils._checks import _check_type
from ..utils._docs import fill_doc


@fill_doc
def calinski_harabasz_score(cluster):  # higher the better
    r"""Compute the Calinski-Harabasz score.

    This function computes the Calinski-Harabasz score\ :footcite:p:`Calinski-Harabasz`
    with :func:`sklearn.metrics.calinski_harabasz_score` from a fitted :ref:`Clustering`
    instance.

    Parameters
    ----------
    %(cluster)s

    Returns
    -------
    score : float
        The resulting Calinski-Harabasz score.

    Notes
    -----
    For more details regarding the implementation, please refer to
    :func:`sklearn.metrics.calinski_harabasz_score`.

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
    # Align polarities just in case..
    x = cluster.cluster_centers_[labels].T
    sign = np.sign((x.T * data.T).sum(axis=1))
    data = data * sign
    score = _calinski_harabasz_score(data.T, labels)
    return score


def _calinski_harabasz_score(X, labels):
    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    n_samples, _ = X.shape
    n_labels = len(le.classes_)

    extra_disp, intra_disp = 0.0, 0.0
    mean = np.mean(X, axis=0)
    for k in range(n_labels):
        cluster_k = X[labels == k]
        mean_k = np.mean(cluster_k, axis=0)
        extra_disp += len(cluster_k) * _distance_matrix([mean_k], [mean])[-1, 0]
        intra_disp += np.sum(_distance_matrix(cluster_k, [mean_k])[-1:, :-1])

    return (
        1.0
        if intra_disp == 0.0
        else extra_disp * (n_samples - n_labels) / (intra_disp * (n_labels - 1.0))
    )
