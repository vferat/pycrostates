"""Calinski Harabasz score."""

import numpy as np
from sklearn.metrics import calinski_harabasz_score as sk_calinski_harabasz_score

from ..cluster._base import _BaseCluster
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
    score = sk_calinski_harabasz_score(data.T, labels)
    return score
