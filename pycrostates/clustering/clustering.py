import os
import itertools

import numpy as np
import scipy
from scipy.signal import find_peaks

import matplotlib.pyplot as plt

import mne
from mne.utils import logger, warn, verbose, _validate_type
from mne.preprocessing.ica import _check_start_stop
from mne.io import BaseRaw
from mne.parallel import check_n_jobs, parallel_func


def _extract_gfps(data, min_peak_dist=2):
    """ Extract Gfp peaks from input data
    Parameters
    ----------
    min_peak_dist : Required minimal horizontal distance (>= 1)
                    in samples between neighbouring peaks.
                    Smaller peaks are removed first until the
                    condition is fulfilled for all remaining peaks.
                    Default to 2.
    X : array-like, shape [n_channels, n_samples]
                The data to extrat Gfp peaks, row by row. scipy.sparse matrices should be
                in CSR format to avoid an un-necessary copy.

    """
    gfp = np.std(data, axis=0)
    peaks, _ = find_peaks(gfp, distance=min_peak_dist)
    gfp_peaks = data[:, peaks]
    return(gfp_peaks)

@verbose
def _compute_maps(data, n_states=4, max_iter=1000, thresh=1e-6,
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
    for _ in range(max_iter):
        # Assign each sample to the best matching microstate
        activation = maps.dot(data)
        segmentation = np.argmax(np.abs(activation), axis=0)

        # Recompute the topographic maps of the microstates, based on the
        # samples that were assigned to each state.
        for state in range(n_states):
            idx = (segmentation == state)
            if np.sum(idx) == 0:
                logger.info('Some microstates are never activated')
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
            #logger.info('Converged at %d iterations.' % iteration)
            break

        prev_residual = residual
    else:
         logger.info('Modified K-means algorithm failed to converge.')

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

def segment(raw, states, half_window_size = 3, factor = 10, crit = 10e-6):
    data = raw.get_data()
    S0 = 0  
    states = (states.T / np.linalg.norm(states, axis=1)).T
    data = (data.T / np.linalg.norm(data, axis=1)).T
    Ne, Nt = data.shape
    Nu = states.shape[0]
    Vvar = np.sum(data * data, axis=0)
    rmat = np.tile(np.arange(0,Nu), (Nt, 1)).T
    
    labels_all = np.argmax(np.abs(np.dot(states, data)), axis=0)
    
    w = np.zeros((Nu,Nt))
    w[(rmat == labels_all)] = 1
    e = np.sum(Vvar - np.sum(np.dot(w.T, states).T * data, axis=0) **2 / (Nt * (Ne - 1)))
    
    window = np.ones((1, 2*half_window_size+1))
    while True:
        Nb = scipy.signal.convolve2d(w, window, mode='same')
        x = (np.tile(Vvar,(Nu,1)) - (np.dot(states, data))**2) / (2* e * (Ne-1)) - factor * Nb   
        dlt = np.argmin(x, axis=0)
        
        labels_all = dlt
        w = np.zeros((Nu,Nt))
        w[(rmat == labels_all)] = 1
        Su = np.sum(Vvar - np.sum(np.dot(w.T, states).T * data, axis=0) **2) / (Nt * (Ne - 1))
        if np.abs(Su - S0) <= np.abs(crit * Su):
            break
        else:
            S0 = Su
    labels = labels_all +1
    return(labels)


class mod_Kmeans():
    @verbose
    def __init__(self, n_clusters: int = 4,
                 random_state=None,
                 n_init : int = 100,
                 max_iter: int = 300,
                 tol: float = 1e-6,
                 verbose = None):
        self.current_fit = False
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        self.GEV = None
        self.cluster_centers = None
        self.labels = None

    def __repr__(self):
        if self.current_fit == True:
            f = 'unfitted'
        else:
            f = 'fitted (raw)'
        s += f' {str(self.n_clusters)}'
        return(f'mod_Kmeans | {s}')
        
    def _run_mod_kmeans(self, data):
        gfp_sum_sq = np.sum(data ** 2)
        maps = _compute_maps(data, self.n_clusters, max_iter=self.max_iter, thresh=self.tol, verbose=self.verbose)
        activation = maps.dot(data)
        segmentation = np.argmax(np.abs(activation), axis=0)
        map_corr = _corr_vectors(data , maps[segmentation].T)
        # Compare across iterations using global explained variance (GEV)
        gev = np.sum((data * map_corr) ** 2) / gfp_sum_sq
        return(gev, maps, segmentation)

    @verbose
    def fit(self, raw : mne.io.RawArray, start : float = None, stop : float = None,
            reject_by_annotation : str = None, gfp : bool = False, n_jobs: int = 1,
            verbose = None):
        _validate_type(raw, (BaseRaw), 'raw', 'Raw')
        reject_by_annotation = 'omit' if reject_by_annotation else None
        start, stop = _check_start_stop(raw, start, stop)
        n_jobs = check_n_jobs(n_jobs)
        
        if len(raw.info['bads']) is not 0:
            warn('Bad channels are present in the recording.',
                  'They will stillbe used to compute microstate topographies.',
                  'Consider using Raw.pick or Raw.interpolate_bads before fitting.')

        data = raw.get_data(start, stop, reject_by_annotation)
        if gfp is True:
            data = _extract_gfps(data)
   
        best_gev = 0
        if n_jobs == 1:
            for _ in range(self.n_init):
                gev, maps, segmentation = self._run_mod_kmeans(data)
                if gev > best_gev:
                    best_gev, best_maps, best_segmentation = gev, maps, segmentation
        else:
            parallel, p_fun, _ = parallel_func(self._run_mod_kmeans, total=self.n_init, n_jobs=n_jobs)
            runs = parallel(p_fun(data) for i in range(self.n_init))
            runs = np.array(runs)
            best_run = np.argmax(runs[:, 0])
            best_gev, best_maps, best_segmentation = runs[best_run]

        self.cluster_centers = best_maps
        self.GEV = best_gev
        self.labels = best_segmentation
        self.current_fit = True
        return(self)
    
    @verbose
    def predict(self, raw : mne.io.RawArray,
                half_window_size: int = 3, factor: int = 10, 
                crit: float = 10e-6,
                verbose = None):
        if self.current_fit == False:
            raise ValueError('mod_Kmeans is not fitted.')
        return(segment(raw, self.cluster_centers, half_window_size, factor, crit))

    def plot_topographies(self, info : mne.Info):
        if self.current_fit == False:
            raise ValueError('mod_Kmeans is not fitted.')
        fig, axs = plt.subplots(1, self.n_clusters)
        for c,center in enumerate(self.cluster_centers):
            mne.viz.plot_topomap(center, info, axes=axs[c], show=False)  
        plt.axis('off')
        plt.show()
        

if __name__ == "__main__":
    from mne.datasets import sample
    import mne
    data_path = sample.data_path()
    raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
    event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'

    # Setup for reading the raw data
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    raw = raw.pick('eeg')
    raw = raw.filter(0,40)
    raw = raw.crop(0,60)
    
    modK = mod_Kmeans()
    modK.fit(raw, gfp=True, n_jobs=5)
    print(modK.GEV)