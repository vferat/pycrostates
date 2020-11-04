import os
import numpy as np
import itertools
import warnings
from mne.utils import logger, verbose
from joblib import Parallel, delayed


def _mod_kmeans(data, n_states=4, n_inits=10, max_iter=1000, thresh=1e-6,
                random_state=None, verbose=None):
    """The modified K-means clustering algorithm.
    See :func:`segment` for the meaning of the parameters and return
    values.
    """
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)
    n_channels, n_samples = data.shape

    # Cache this value for later
    data_sum_sq = np.sum(data ** 2)

    # Select random timepoints for our initial topographic maps
    init_times = random_state.choice(n_samples, size=n_states, replace=False)
    maps = data[:, init_times].T
    maps /= np.linalg.norm(maps, axis=1, keepdims=True)  # Normalize the maps

    prev_residual = np.inf
    for iteration in range(max_iter):
        # Assign each sample to the best matching microstate
        activation = maps.dot(data)
        segmentation = np.argmax(np.abs(activation), axis=0)

        # Recompute the topographic maps of the microstates, based on the
        # samples that were assigned to each state.
        for state in range(n_states):
            idx = (segmentation == state)
            if np.sum(idx) == 0:
                warnings.warn('Some microstates are never activated')
                maps[state] = 0
                continue
            # Find largest eigenvector
            # cov = data[:, idx].dot(data[:, idx].T)
            # _, vec = eigh(cov, eigvals=(n_channels - 1, n_channels - 1))
            # maps[state] = vec.ravel()
            maps[state] = data[:, idx].dot(activation[state, idx])
            maps[state] /= np.linalg.norm(maps[state])

        # Estimate residual noise
        act_sum_sq = np.sum(np.sum(maps[segmentation].T * data, axis=0) ** 2)
        residual = abs(data_sum_sq - act_sum_sq)
        residual /= float(n_samples * (n_channels - 1))

        # Have we converged?
        if (prev_residual - residual) < (thresh * residual):
            logger.info('Converged at %d iterations.' % iteration)
            break

        prev_residual = residual
    else:
        warnings.warn('Modified K-means algorithm failed to converge.')

    return maps


def _corr_vectors(A, B, axis=0):
    """Compute pairwise correlation of multiple pairs of vectors.
    Fast way to compute correlation of multiple pairs of vectors without
    computing all pairs as would with corr(A,B). Borrowed from Oli at Stack
    overflow. Note the resulting coefficients vary slightly from the ones
    obtained from corr due differences in the order of the calculations.
    (Differences are of a magnitude of 1e-9 to 1e-17 depending of the tested
    data).
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
    corr : ndarray, shape (m,)
        For each pair of vectors, the correlation between them.
    """
    An = A - np.mean(A, axis=axis)
    Bn = B - np.mean(B, axis=axis)
    An /= np.linalg.norm(An, axis=axis)
    Bn /= np.linalg.norm(Bn, axis=axis)
    return np.sum(An * Bn, axis=axis)


class mod_Kmeans():
    def __init__(self, n_clusters=4, n_init=100,
             max_iter=300, tol=1e-6,
             verbose=None, random_state=None,
             n_jobs=None):

        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.GEV = None
        self.cluster_centers = None
        self.labels = None

    def _run_mod_kmeans(self, X):
        gfp_sum_sq = np.sum(X ** 2)
        maps = _mod_kmeans(X, self.n_clusters, n_inits=None, max_iter=self.max_iter, thresh=self.tol, verbose=self.verbose)
        activation = maps.dot(X)
        segmentation = np.argmax(np.abs(activation), axis=0)
        map_corr = _corr_vectors(X , maps[segmentation].T)
        # Compare across iterations using global explained variance (GEV)
        gev = np.sum((X * map_corr) ** 2) / gfp_sum_sq
        return(gev, maps, segmentation)

    def fit(self, X):
        best_gev = 0

        if self.n_jobs is not None:
            runs = Parallel(n_jobs=self.n_jobs)(delayed(self._run_mod_kmeans)(X) for i in range(self.n_init))
            runs = np.array(runs)
            best_run = np.argmax(runs[:, 0])
            best_gev, best_maps, best_segmentation = runs[best_run]
        else:
            gfp_sum_sq = np.sum(X ** 2)
            for _ in range(self.n_init):
                maps = _mod_kmeans(X, self.n_clusters, n_inits=None, max_iter=self.max_iter, thresh=self.tol, verbose=self.verbose)
                activation = maps.dot(X)
                segmentation = np.argmax(np.abs(activation), axis=0)
                map_corr = _corr_vectors(X , maps[segmentation].T)
                # Compare across iterations using global explained variance (GEV)
                gev = np.sum((X * map_corr) ** 2) / gfp_sum_sq
                if gev > best_gev:
                    best_gev, best_maps, best_segmentation = gev, maps, segmentation

        self.cluster_centers = best_maps
        self.GEV = best_gev
        self.labels = best_segmentation
        return(self)
