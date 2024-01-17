from ..cluster._base import _BaseCluster as _BaseCluster
from ..utils._checks import _check_type as _check_type
from ..utils._docs import fill_doc as fill_doc

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
    :func:`sklearn.metrics.calinski_harabasz_score`.

    References
    ----------
    .. footbibliography::
    """
