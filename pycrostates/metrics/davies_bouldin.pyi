from ..cluster._base import _BaseCluster as _BaseCluster
from ..utils import _distance_matrix as _distance_matrix
from ..utils._checks import _check_type as _check_type
from ..utils._docs import fill_doc as fill_doc

def davies_bouldin_score(cluster):
    """Compute the Davies-Bouldin score.

    This function computes the Davies-Bouldin score\\ :footcite:p:`Davies-Bouldin` with
    :func:`sklearn.metrics.davies_bouldin_score` from a fitted :ref:`Clustering`
    instance.

    Parameters
    ----------
    cluster : :ref:`cluster`
        Fitted clustering algorithm from which to compute score. For more details about
        current clustering implementations, check the :ref:`Clustering` section of the
        documentation.

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
