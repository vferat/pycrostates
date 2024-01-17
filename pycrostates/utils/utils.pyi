from _typeshed import Incomplete

from ._logs import logger as logger

def _corr_vectors(A, B, axis: int = 0):
    """Compute pairwise correlation of multiple pairs of vectors.

    Fast way to compute correlation of multiple pairs of vectors without computing all
    pairs as would with corr(A,B). Borrowed from Oli at StackOverflow. Note the
    resulting coefficients vary slightly from the ones obtained from corr due to
    differences in the order of the calculations. (Differences are of a magnitude of
    1e-9 to 1e-17 depending on the tested data).

    Parameters
    ----------
    A : ndarray, shape (n, m)
        The first collection of vectors
    B : ndarray, shape (n, m)
        The second collection of vectors
    axis : int
        The axis that contains the elements of each vector. Defaults to 0.

    Returns
    -------
    corr : ndarray, shape (m, )
        For each pair of vectors, the correlation between them.
    """

def _distance_matrix(X, Y: Incomplete | None = None):
    """Distance matrix used in metrics."""

def _compare_infos(cluster_info, inst_info):
    """Check that channels in cluster_info are all present in inst_info."""
