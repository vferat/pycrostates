from ..cluster._base import _BaseCluster as _BaseCluster
from ..utils import _distance_matrix as _distance_matrix
from ..utils._checks import _check_type as _check_type
from ..utils._docs import fill_doc as fill_doc

def silhouette_score(cluster):
    """Compute the mean Silhouette Coefficient.

    This function computes the Silhouette Coefficient\\ :footcite:p:`Silhouettes` with
    :func:`sklearn.metrics.silhouette_score` from a fitted :ref:`Clustering` instance.

    Parameters
    ----------
    cluster : :ref:`cluster`
        Fitted clustering algorithm from which to compute score. For more details about
        current clustering implementations, check the :ref:`Clustering` section of the
        documentation.

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
