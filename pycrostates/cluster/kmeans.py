from ._base import _BaseCluster
from ..utils._checks import _check_type
from ..utils._docs import fill_doc, copy_doc
from ..utils._logs import _set_verbose


@fill_doc
class KMeans(_BaseCluster):
    """
    K-Means clustering algorithms.

    Parameters
    ----------
    n_clusters : int
        The number of clusters to form as well as the number of centroids to
        generate.
    n_init : int
        Number of time the k-means algorithm is run with different centroid
        seeds. The final result will be the run with highest global explained
        variance.
    max_iter : int
        Maximum number of iterations of the k-means algorithm for a single run.
    tol : float
        Relative tolerance with regards estimate residual noise in the cluster
        centers of two consecutive iterations to declare convergence.
    %(random_state)s
    """
    # TODO: docstring for tol doesn't look english

    def __init__(self, n_clusters, n_init=100, max_iter=300, tol=1e-6,
                 random_state=None):
        super().__init__()

        # k-means has a fix number of clusters defined at init
        self._n_clusters = _BaseCluster._check_n_clusters(n_clusters)
        self._clusters_names = [str(k) for k in range(1, self.n_clusters + 1)]

        # k-means settings
        self._n_init = KMeans._check_n_init(n_init)
        self._max_iter = KMeans._check_max_iter(max_iter)
        self._tol = KMeans._check_tol(tol)
        self._random_state = random_state

    @copy_doc(_BaseCluster.fit)
    @fill_doc
    def fit(self, inst, picks='eeg', tmin=None, tmax=None,
            reject_by_annotation=True, n_jobs=1, *, verbose=None):
        """
        %(verbose)s
        """
        _set_verbose(verbose)  # TODO: decorator nesting is failing
        super().fit(inst, picks, tmin, tmax, reject_by_annotation, n_jobs)

    # --------------------------------------------------------------------
    @property
    def n_init(self):
        """
        Number of time the k-means algorithm is run with different centroid
        seeds.

        :type: `int`
        """
        return self._n_init

    @property
    def max_iter(self):
        """
        Number of maximum iterations of the k-means algorithm for a single run.

        :type: `int`
        """
        return self._max_iter

    @property
    def tol(self):
        """
        Relative tolerance.

        :type: `float`
        """
        return self._tol

    @property
    def random_state(self):
        """
        Random state.

        :type: `None` | `int` | `~numpy.random.RandomState`
        """
        return self._random_state

    # --------------------------------------------------------------------
    # ---------------
    # For now, check function are defined as static within KMeans. If they are
    # used outside KMeans, they should be moved to regular function in _base.py
    # ---------------
    @staticmethod
    def _check_n_init(n_init: int):
        """Check that n_init is a positive integer."""
        _check_type(n_init, ('int', ), item_name='n_init')
        if n_init <= 0:
            raise ValueError(
                "The number of initialization must be a positive integer. "
                f"Provided: '{n_init}'.")
        return n_init

    @staticmethod
    def _check_max_iter(max_iter: int):
        """Check that max_iter is a positive integer."""
        _check_type(max_iter, ('int', ), item_name='max_iter')
        if max_iter <= 0:
            raise ValueError(
                "The number of max iteration must be a positive integer. "
                f"Provided: '{max_iter}'.")
        return max_iter

    @staticmethod
    def _check_tol(tol: float):
        """Check that tol is a positive number."""
        _check_type(tol, ('numeric', ), item_name='tol')
        if tol <= 0:
            raise ValueError(
                "The tolerance must be a positive number. "
                f"Provided: '{tol}'.")
        return tol
