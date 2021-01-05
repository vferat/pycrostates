from __future__ import annotations

import os
import itertools

import numpy as np
import scipy
from scipy.signal import find_peaks

import matplotlib
import matplotlib.pyplot as plt

import mne
from mne.utils import logger, warn, verbose, _validate_type
from mne.preprocessing.ica import _check_start_stop
from mne.annotations import _annotations_starts_stops
from mne.io import BaseRaw
from mne.parallel import check_n_jobs, parallel_func

from typing import Tuple, Union


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
            # logger.info('Converged at %d iterations.' % iteration)
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

def _segment(data, states, half_window_size=3, factor=10, crit=10e-6):
    S0 = 0
    states = (states.T / np.std(states, axis=1)).T
    data = (data.T / np.std(data, axis=1)).T
    Ne, Nt = data.shape
    Nu = states.shape[0]
    Vvar = np.sum(data * data, axis=0)
    rmat = np.tile(np.arange(0, Nu), (Nt, 1)).T

    labels_all = np.argmax(np.abs(np.dot(states, data)), axis=0)

    w = np.zeros((Nu, Nt))
    w[(rmat == labels_all)] = 1
    e = np.sum(Vvar - np.sum(np.dot(w.T, states).T *
                             data, axis=0) ** 2 / (Nt * (Ne - 1)))

    window = np.ones((1, 2*half_window_size+1))
    while True:
        Nb = scipy.signal.convolve2d(w, window, mode='same')
        x = (np.tile(Vvar, (Nu, 1)) - (np.dot(states, data))**2) / \
            (2 * e * (Ne - 1)) - factor * Nb
        dlt = np.argmin(x, axis=0)

        labels_all = dlt
        w = np.zeros((Nu, Nt))
        w[(rmat == labels_all)] = 1
        Su = np.sum(Vvar - np.sum(np.dot(w.T, states).T *
                                  data, axis=0) ** 2) / (Nt * (Ne - 1))
        if np.abs(Su - S0) <= np.abs(crit * Su):
            break
        else:
            S0 = Su

    labels = labels_all + 1
    # set first segment to unlabeled
    i = 0
    first_label = labels[i]
    while labels[i] == first_label and i < len(labels) - 1:
        labels[i] = 0
        i += 1
    # set last segment to unlabeled
    i = len(labels) - 1
    last_label = labels[i]
    while labels[i] == last_label and i > 0:
        labels[i] = 0
        i -= 1
    return(labels)

class BaseClustering():
    def __init__(self, n_clusters: int = 4,
                 verbose=None): 
        self.n_clusters = n_clusters
        self.verbose = verbose
        self.current_fit = False
        self.GEV = None
        self.cluster_centers = None
        self.Info = None

    def __repr__(self) -> str:
        if self.current_fit is False:
            s = '| unfitted'
        else:
            s = '| fitted (raw)'
        s = f' n = {str(self.n_clusters)} cluster centers ' + s
        return(f'{self.__class__.__name__} | {s}')

    @verbose
    def fit(self, raw: mne.io.RawArray, verbose=None, *args, **kwargs):
        _validate_type(raw, (BaseRaw), 'raw', 'Raw')
        cluster_centers, GEV, labels =  self._do_fit(raw, verbose=verbose, *args, **kwargs)
        self.cluster_centers = cluster_centers
        self.GEV = GEV
        self.current_fit = True
        self.info = raw.info
        return()

    @verbose 
    def transform(self, raw: mne.io.RawArray) -> np.ndarray:
        data = raw.get_data()
        stack = np.vstack([self.cluster_centers, data.T])
        corr = np.corrcoef(stack)[:self.n_clusters, self.n_clusters:]
        distances = np.max(np.abs(corr), axis=0)
        return(distances)

    @verbose
    def predict(self, raw: mne.io.RawArray,
                reject_by_annotation: bool = True,
                half_window_size: int = 3, factor: int = 10,
                crit: float = 10e-6,
                verbose: str = None) -> np.ndarray:
        """[summary]

        Args:
            raw (mne.io.RawArray): [description]
            reject_by_annotation (bool, optional): [description]. Defaults to True.
            half_window_size (int, optional): [description]. Defaults to 3.
            factor (int, optional): [description]. Defaults to 10.
            crit (float, optional): [description]. Defaults to 10e-6.
            verbose (str, optional): [description]. Defaults to None.

        Raises:
            ValueError: [description]

        Returns:
            np.ndarray: [description]
        """
        if self.current_fit is False:
            raise ValueError('mod_Kmeans is not fitted.')

        data = raw.get_data()
        if reject_by_annotation:
            onsets, _ends = _annotations_starts_stops(raw, ['BAD'])
            if len(onsets) == 0:
                return(_segment(data, self.cluster_centers,
                               half_window_size, factor, crit))

            onsets = onsets.tolist()
            onsets.append(data.shape[-1] - 1)
            _ends = _ends.tolist()
            ends = [0]
            ends.extend(_ends)
            segmentation = np.zeros(data.shape[-1])
            for onset, end in zip(onsets, ends):
                if onset - end >= 2 * half_window_size + 1:  # small segments can't be smoothed
                    sample = data[:, end:onset]
                    segmentation[end:onset] = _segment(sample,
                                                       self.cluster_centers,
                                                       half_window_size, factor,
                                                       crit)
            return(segmentation)

        else:
            return(_segment(data,
                            self.cluster_centers,
                            half_window_size, factor,
                            crit))       

    def plot_cluster_centers(self, info: mne.Info = None) -> Tuple[matplotlib.figure.Figure,
                                                                   matplotlib.axes.Axes]:
        """[summary]

        Args:
            info (mne.Info): [description]

        Raises:
            ValueError: [description]

        Returns:
            Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: [description]
        """

        if self.current_fit is False:
            raise ValueError('mod_Kmeans is not fitted.')
        if not info:
            info = self.info
        fig, axs = plt.subplots(1, self.n_clusters)
        for c, center in enumerate(self.cluster_centers):
            mne.viz.plot_topomap(center, info, axes=axs[c], show=False)
        plt.axis('off')
        plt.show()
        return(fig, axs)

    def reorder(self, order: list):
        if (np.sort(order) != np.arange(0,self.n_clusters, 1)).any():
            raise ValueError('Order contains unexpected values')
        else:
            self.cluster_centers = self.cluster_centers[order]
        return(self)

    def smart_reorder(self):
        if self.current_fit is False:
            raise ValueError('mod_Kmeans is not fitted.')
        info = self.info
        centers = self.cluster_centers
        
        template = np.array([[-0.13234463, -0.19008217, -0.01808156, -0.06665204, -0.18127315,
        -0.25741473, -0.2313206 ,  0.04239534, -0.14411298, -0.25635016,
         0.1831745 ,  0.17520883, -0.06034687, -0.21948988, -0.2057277 ,
         0.27723199,  0.04632557, -0.1383458 ,  0.36954792,  0.33889126,
         0.1425386 , -0.05140216, -0.07532628,  0.32313928,  0.21629226,
         0.11352515],
       [-0.15034466, -0.08511373, -0.19531161, -0.24267313, -0.16871454,
        -0.04761393,  0.02482456, -0.26414511, -0.15066143,  0.04628036,
        -0.1973625 , -0.24065874, -0.08569745,  0.1729162 ,  0.22345117,
        -0.17553494,  0.00688743,  0.25853483, -0.09196588, -0.09478585,
         0.09460047,  0.32742083,  0.4325027 ,  0.09535141,  0.1959104 ,
         0.31190313],
       [ 0.29388541,  0.2886461 ,  0.27804376,  0.22674127,  0.21938115,
         0.21720292,  0.25153101,  0.12125869,  0.10996983,  0.10638135,
         0.11575272, -0.01388831, -0.04507772, -0.03708886,  0.08203929,
        -0.14818182, -0.20299531, -0.16658826, -0.09488949, -0.23512102,
        -0.30464665, -0.25762648, -0.14058166, -0.22072284, -0.22175042,
        -0.22167467],
       [-0.21660409, -0.22350361, -0.27855619, -0.0097109 ,  0.07119601,
         0.00385336, -0.24792901,  0.08145982,  0.23290418,  0.09985582,
        -0.24242583,  0.13516244,  0.3304661 ,  0.16710186, -0.21832217,
         0.15575575,  0.33346027,  0.18885162, -0.21687347,  0.10926662,
         0.26182733,  0.13760157, -0.19536083, -0.15966419, -0.14684497,
        -0.15296749],
       [-0.12444958, -0.12317709, -0.06189361, -0.20820917, -0.25736043,
        -0.20740485, -0.06941215, -0.18086612, -0.26979589, -0.17602898,
         0.05332203, -0.10101208, -0.20095764, -0.09582802,  0.06883067,
         0.0082463 , -0.07052899,  0.00917889,  0.26984673,  0.13288481,
         0.08062487,  0.13616082,  0.30845643,  0.36843231,  0.35510687,
         0.35583386]])
        ch_names_template =  ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC3', 'FCz',
                            'FC4', 'T3', 'C3', 'Cz', 'C4', 'T4', 'CP3', 'CPz', 'CP4',
                            'T5', 'P3', 'Pz', 'P4','T6', 'O1', 'Oz', 'O2']
        
        ch_names_template = [name.lower() for name in ch_names_template]
        ch_names_centers = [name.lower() for name in info['ch_names']]
        common_ch_names = list(set(ch_names_centers).intersection(ch_names_template))
        
        if len (common_ch_names) <= 10:
            warn("Not enought common electrodes with built-in template to automaticalv reorder maps. "
                 "Order hasn't been changed.")
            return()
        
        common_names_template = [ch_names_template.index(name) for name in common_ch_names]
        common_names_centers = [ch_names_centers.index(name) for name in common_ch_names]

        reduc_template = template[:, common_names_template]
        reduc_centers = centers[:, common_names_centers]
          
        mat = np.corrcoef(reduc_template,reduc_centers)[:len(reduc_template), -len(reduc_centers):]
        mat = np.abs(mat)
        mat_ = mat.copy()
        rows = list()
        columns = list()
        while len(columns) < len(template) and len(columns) < len(centers):
            mask_columns = np.ones(mat.shape[1], bool)
            mask_rows = np.ones(mat.shape[0], bool)
            mask_rows[rows] = 0
            mask_columns[columns] = 0
            mat_ = mat[mask_rows,:][:,mask_columns]
            row, column = np.unravel_index(np.where(mat.flatten() == np.max(mat_))[0][0], mat.shape)
            rows.append(row)
            columns.append(column)
            mat[row, column] = -1
        order = [x for _,x in sorted(zip(rows,columns))]
        order = order + [x for x in range(len(centers)) if x not in order]
        self.cluster_centers = centers[order]
        return()

    def _do_fit(self, raw, *args, **kwargs):
        """Function to overwrite"""
        return()

class ModKMeans(BaseClustering):
    def __init__(self,
                 random_state: Union[float, np.random.RandomState] = None,
                 n_init: int = 100,
                 max_iter: int = 300,
                 tol: float = 1e-6,
                 *args,  **kwargs):
        super().__init__(*args, **kwargs)
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.labels = None

    def _run_mod_kmeans(self, data: np.ndarray) -> Tuple[float,
                                                         np.ndarray,
                                                         np.ndarray]:
        gfp_sum_sq = np.sum(data ** 2)
        maps = _compute_maps(data, self.n_clusters, max_iter=self.max_iter,
                             random_state=self.random_state,
                             thresh=self.tol, verbose=self.verbose)
        activation = maps.dot(data)
        segmentation = np.argmax(np.abs(activation), axis=0)
        map_corr = _corr_vectors(data, maps[segmentation].T)
        # Compare across iterations using global explained variance (GEV)
        gev = np.sum((data * map_corr) ** 2) / gfp_sum_sq
        return(gev, maps, segmentation)

    def _do_fit(self, raw: mne.io.RawArray, start: float = None, stop: float = None,
            reject_by_annotation: bool = True,
            gfp: bool = False, n_jobs: int = 1,
            verbose=None) -> ModKMeans:

        reject_by_annotation = 'omit' if reject_by_annotation else None
        start, stop = _check_start_stop(raw, start, stop)
        n_jobs = check_n_jobs(n_jobs)

        if len(raw.info['bads']) is not 0:
            warn('Bad channels are present in the recording. '
                 'They will still be used to compute microstate topographies. '
                 'Consider using Raw.pick() or Raw.interpolate_bads()'
                 ' before fitting.')

        data = raw.get_data(start, stop,
                            reject_by_annotation=reject_by_annotation)
        if gfp is True:
            data = _extract_gfps(data)

        best_gev = 0
        if n_jobs == 1:
            for _ in range(self.n_init):
                gev, maps, segmentation = self._run_mod_kmeans(data)
                if gev > best_gev:
                    best_gev, best_maps, best_segmentation = gev, maps, segmentation
        else:
            parallel, p_fun, _ = parallel_func(self._run_mod_kmeans,
                                               total=self.n_init,
                                               n_jobs=n_jobs)
            runs = parallel(p_fun(data) for i in range(self.n_init))
            runs = np.array(runs)
            best_run = np.argmax(runs[:, 0])
            best_gev, best_maps, best_segmentation = runs[best_run]

        return(best_maps, best_gev, best_segmentation)

