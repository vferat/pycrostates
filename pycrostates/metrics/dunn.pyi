from ..cluster._base import _BaseCluster as _BaseCluster
from ..utils import _distance_matrix as _distance_matrix
from ..utils._checks import _check_type as _check_type
from ..utils._docs import fill_doc as fill_doc

def dunn_score(cluster):
    """Compute the Dunn index score.

    This function computes the Dunn index score\\ :footcite:p:`Dunn` from a
    fitted :ref:`Clustering` instance.

    Parameters
    ----------
    cluster : :ref:`cluster`
        Fitted clustering algorithm from which to compute score. For more details about
        current clustering implementations, check the :ref:`Clustering` section of the
        documentation.

    Returns
    -------
    score : float
        The resulting Dunn score.

    Notes
    -----
    This function uses the absolute spatial correlation for distance.

    References
    ----------
    .. footbibliography::
    """

def _dunn_score(X, labels):
    """Compute the Dunn index.

    Parameters
    ----------
    X : np.array
        np.array([N, p]) of all points
    labels: np.array
        np.array([N]) labels of all points

    Notes
    -----
    Based on https://github.com/jqmviegas/jqm_cvi
    """

def _delta_fast(ck, cl, distances): ...
def _big_delta_fast(ci, distances): ...
