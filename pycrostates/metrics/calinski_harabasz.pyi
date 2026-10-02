from ..cluster._base import _BaseCluster as _BaseCluster
from ..utils import _distance_matrix as _distance_matrix
from ..utils._checks import _check_type as _check_type
from ..utils._docs import fill_doc as fill_doc

@fill_doc
def calinski_harabasz_score(cluster):
    """Compute the Calinski-Harabasz score.

    This function computes the Calinski-Harabasz score\\ :footcite:p:`Calinski-Harabasz`
    with :func:`sklearn.metrics.calinski_harabasz_score` from a fitted :ref:`Clustering`
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
        The resulting Calinski-Harabasz score.

    Notes
    -----
    For more details regarding the implementation, please refer to
    :func:`sklearn.metrics.calinski_harabasz_score`. This implementation is modified
    to use absolute spatial correlation for distance computations instead of the
    Euclidean distance.

    References
    ----------
    .. footbibliography::
    """

def _calinski_harabasz_score(X, labels): ...
