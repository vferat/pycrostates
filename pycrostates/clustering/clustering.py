from __future__ import annotations
import itertools
from copy import deepcopy
from pycrostates.viz import plot_cluster_centers

from typing import Tuple

import matplotlib
import mne
import numpy as np
import scipy

from mne.annotations import _annotations_starts_stops
from mne.io import BaseRaw
from mne.epochs import BaseEpochs
from mne.parallel import check_n_jobs, parallel_func
from mne.preprocessing.ica import _check_start_stop
from mne.utils import _validate_type, logger, verbose, warn, fill_doc, check_random_state
from mne.io.pick import _picks_to_idx, pick_info

from ..utils import _corr_vectors, _check_ch_names, _check_reject_by_annotation
from ..segmentation import RawSegmentation, EpochsSegmentation

@verbose
def _compute_maps(data, n_states=4, max_iter=1000, tol=1e-6,
                  random_state=None, verbose=None):
    # based on https://github.com/wmvanvliet/mne_microstates/blob/master/microstates.py
    # written by Marijn van Vliet <w.m.vanvliet@gmail.com>

    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)
    # handle zeros maps
    # zero map can be due to non data in the recording, it's unlikly that all channels recorded the same value at the same time (=0 due to avergare reference)
    data = data[:, np.linalg.norm(data.T, axis=1) !=0]
    n_channels, n_samples = data.shape
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
                #logger.info('Some microstates are never activated')
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
        if (prev_residual - residual) < (tol * residual):
            # logger.info('Converged at %d iterations.' % iteration)
            break

        prev_residual = residual
    else:
        logger.info('Modified K-means algorithm failed to converge.')

    return maps

@verbose
def _run_mod_kmeans(data: np.ndarray, n_clusters=4,
                    max_iter=100, random_state=None,
                    tol = 1e-6, verbose=None) -> Tuple[float,
                                                        np.ndarray,
                                                        np.ndarray]:
    gfp_sum_sq = np.sum(data ** 2)
    maps = _compute_maps(data, n_clusters, max_iter=max_iter,
                            random_state=random_state,
                            tol=tol, verbose=verbose)
    activation = maps.dot(data)
    segmentation = np.argmax(np.abs(activation), axis=0)
    map_corr = _corr_vectors(data, maps[segmentation].T)
    # Compare across iterations using global explained variance (GEV)
    gev = np.sum((data * map_corr) ** 2) / gfp_sum_sq
    return(gev, maps, segmentation)

def _segment(data, states, half_window_size=3, factor=0, crit=10e-6):
    S0 = 0
    states = (states.T / np.std(states, axis=1)).T
    data = (data.T / np.std(data, axis=1)).T
    Ne, Nt = data.shape
    Nu = states.shape[0]
    Vvar = np.sum(data * data, axis=0)
    rmat = np.tile(np.arange(0, Nu), (Nt, 1)).T

    labels_all = np.argmax(np.abs(np.dot(states, data)), axis=0)

    if factor != 0:
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
            S0 = Su

    labels = labels_all + 1
    return(labels)


def _rejected_first_last_segments(segmentation):
    # set first segment to unlabeled
    i = 0
    first_label = segmentation[i]
    if  first_label != 0:
        while segmentation[i] == first_label and i < len(segmentation) - 1:
            segmentation[i] = 0
            i += 1
    # set last segment to unlabeled
    i = len(segmentation) - 1
    last_label = segmentation[i]
    if  last_label != 0:
        while segmentation[i] == last_label and i > 0:
            segmentation[i] = 0
            i -= 1
    return(segmentation)

def _reject_small_segments(segmentation, data, min_segment_lenght):
    new_segmentation = segmentation.copy()
    small_segment = True
    while small_segment:
        segments = [list(group) for _,group in itertools.groupby(new_segmentation)]
        current_idx = 0
        small_segment = False
        for i,segment in enumerate(segments):
            if i not in [0,len(segments)-1]:
                if len(segment) <= min_segment_lenght and segment[0] != 0:
                    small_segment = True
                    left_idx = current_idx
                    right_idx = current_idx + len(segment)
                    new_segment = new_segmentation[left_idx:right_idx]

                    while len(new_segment) != 0:
                        left_corr = np.abs(_corr_vectors(data[:,left_idx - 1].T, data[:,left_idx].T,))
                        right_corr = np.abs(_corr_vectors(data[:,right_idx + 1].T, data[:,right_idx].T))

                        if left_corr < right_corr:
                            new_segmentation[right_idx-1] = new_segmentation[right_idx]
                            right_idx -= 1
                            new_segment = new_segment[:-1]
                        else:
                            new_segmentation[left_idx] = new_segmentation[left_idx - 1]
                            left_idx += 1
                        new_segment = new_segmentation[left_idx:right_idx]
                    break
            current_idx += len(segment)
    return(new_segmentation)


@fill_doc
class BaseClustering():
    u"""Base Class for Microstate Clustering algorithms.

    Parameters
    ----------
    n_clusters : int
        The number of clusters to form as well as the number of centroids to generate.
    picks : str | list | slice | None
        Channels to include. Slices and lists of integers will be interpreted as channel indices.
        In lists, channel type strings (e.g., ['meg', 'eeg']) will pick channels of those types,
        channel name strings (e.g., ['MEG0111', 'MEG2623'] will pick the given channels.
        Can also be the string values “all” to pick all channels, or “data” to pick data channels.
        None will pick all channels.
        Note that channels in info['bads'] will be included if their names or indices are explicitly provided.
        Default to 'eeg'.

    Attributes
    ----------
    n_clusters : int
        The number of clusters to form as well as the number of centroids to generate.
    current_fit : bool
        Flag informing about which data type (raw, epochs or evoked) was used for the fit.
    cluster_centers_ : :class:`numpy.ndarray`, shape ``(n_clusters, n_channels)``
            Cluster centers (i.e Microstates maps).
    info : dict
            :class:`Measurement info <mne.Info>` of fitted instance.
    picks : str | list | slice | None
            picks.
    """
    def __init__(self, n_clusters=4, picks='eeg'):
        self.n_clusters = n_clusters
        self.picks = picks
        self.current_fit = 'unfitted'
        self.names = [f'{i+1}' for i in range(n_clusters)]
        self.info = None

    def __repr__(self) -> str:
        s = f'{self.__class__.__name__} | n = {str(self.n_clusters)} cluster centers | {self.current_fit}'
        return(s)

    def copy(self):
        """Return a copy of the instance.
        """
        return deepcopy(self)

    def _check_fit(self):
        if self.current_fit == 'unfitted':
            raise ValueError(f'Algorithm must be fitted before using {self.__class__.__name__}')
        return()

    def get_cluster_centers(self):
        """Get cluster centers as a :class:`numpy.ndarray`
        """
        self._check_fit()
        cluster_centers = self.cluster_centers_.copy()
        return(cluster_centers)

    def get_cluster_centers_as_raw(self):
        """Get cluster centers as a :class:`mne.io.Raw`
        """
        self._check_fit()
        cluster_centers_raw = mne.io.RawArray(data=self.cluster_centers_.T, info=self.info)
        return(cluster_centers_raw)

    def invert_polarity(self, invert):
        """Invert map polarities.
           This method is for visualisation purpose only
           and has no impact on further processing as map polarities are ignored.
           Operates in place

        Parameters
        ----------
        invert : list of bool
            List of bool of length n_clusters.
            True will invert map polarity, while False will have no effect.
        """
        self._check_fit()
        cluster_centers = self.cluster_centers_
        for c,cluster in enumerate(cluster_centers):
            if invert[c]:
                cluster_centers[c] = - cluster
        self.cluster_centers_ = cluster_centers
        return(self)


    @verbose
    def _predict_raw(self, raw, picks, reject_by_annotation, half_window_size, factor, crit, min_segment_lenght, rejected_first_last_segments, verbose=None):
        data = raw.get_data(picks=picks)
        if not reject_by_annotation:
            segmentation = _segment(data,
                                self.cluster_centers_,
                                half_window_size, factor,
                                crit)
        else:
            onsets, _ends = _annotations_starts_stops(raw, ['BAD'])
            if len(onsets) == 0:
                segmentation = _segment(data,
                                        self.cluster_centers_,
                                         half_window_size, factor,
                                         crit)
            else:
                onsets = onsets.tolist()
                onsets.append(data.shape[-1] - 1)
                _ends = _ends.tolist()
                ends = [0]
                ends.extend(_ends)

                segmentation = np.zeros(data.shape[-1])
                for onset, end in zip(onsets, ends):
                    if onset - end >= 2 * half_window_size + 1:  # small segments can't be smoothed
                        sample = data[:, end:onset]
                        labels =  _segment(sample,
                                            self.cluster_centers_,
                                            half_window_size, factor,
                                            crit)
                        if rejected_first_last_segments:
                            labels = _rejected_first_last_segments(labels)
                        segmentation[end:onset] = labels

        if min_segment_lenght > 0:
            segmentation = _reject_small_segments(segmentation, data, min_segment_lenght)
        if rejected_first_last_segments:
            segmentation = _rejected_first_last_segments(segmentation)


        segmentation = RawSegmentation(segmentation=segmentation,
                                       inst=raw,
                                       cluster_centers=self.cluster_centers_,
                                       names=self.names)
        return(segmentation)

    @verbose
    def _predict_epochs(self, epochs, picks, half_window_size, factor, crit, min_segment_lenght, rejected_first_last_segments, verbose=None):
        data = epochs.get_data(picks=picks)
        segments = list()
        for epoch in data:
            segment =  _segment(epoch,
                                self.cluster_centers_,
                                half_window_size, factor,
                                crit)

            if min_segment_lenght > 0:
                segment = _reject_small_segments(segment, epoch, min_segment_lenght)

            if rejected_first_last_segments:
                segment = _rejected_first_last_segments(segment)
            segments.append(segment)

        segments = np.array(segments)
        segmentation = EpochsSegmentation(segmentation=segments,
                                          inst=epochs,
                                          cluster_centers=self.cluster_centers_,
                                          names=self.names)
        return(segmentation)

    @fill_doc
    @verbose
    def predict(self,  inst,
                reject_by_annotation=True,
                factor=0,
                half_window_size=3,
                crit=10e-6,
                min_segment_lenght=0,
                rejected_first_last_segments=True,
                verbose=None):
        """Predict Microstates labels using competitive fitting.

        Parameters
        ----------
        inst : :class:`mne.io.BaseRaw`, :class:`mne.Epochs`
            Instance containing data to predict.
        factor: int
            Factor used for label smoothing. 0 means no smoothing.
            Defaults to 0.
        half_window_size: int
            Number of samples used for the half windows size while smoothing labels.
            Has no ffect if factor = 0.
            Window size = 2 * half_window_size + 1
        crit: float
            Converge criterion. Default to 10e-6.
        min_segment_lenght: int
            Minimum segment length (in samples). If a segment is shorter than this value,
            it will be recursively reasigned to neighbouring segments based on absolute spatial correlation.
        rejected_first_last_segments: bool
            If True, set first and last segments to unlabeled.
            Default to True.
        reject_by_annotation : bool
            Whether to reject by annotation. If True (default), segments annotated with description starting with ‘bad’ are omitted.
            If False,no rejection is done.
        %(verbose)s

        Returns
        ----------
        segmentation : :class:`numpy.ndarray`
                Microstate sequence derivated from Instance data. Timepoints are labeled according
                to cluster centers number: 1 for the first center, 2 for the second ect..
                0 is used for unlabeled time points.
        """
        self._check_fit()
        _validate_type(inst, (BaseRaw, BaseEpochs), 'inst', 'Raw or Epochs')
        inst = inst.copy()
        picks = _picks_to_idx(inst.info, self.picks,
                        exclude=[], allow_empty=False)
        inst = inst.pick(picks)

        _check_ch_names(self, inst, inst1_name=type(self).__name__, inst2_name='inst')
        if factor == 0:
            logger.info('Segmenting data without smoothing')
        if factor != 0:
            logger.info(f'Segmenting data with factor {factor} and effective smoothing window size : {(2*half_window_size+1) / inst.info["sfreq"]} (ms)')
        if min_segment_lenght > 0:
            logger.info(f'Rejecting segments shorter than {min_segment_lenght / inst.info["sfreq"]} (ms)')
        if rejected_first_last_segments:
            logger.info('Rejecting first and last segment')

        if isinstance(inst, BaseRaw):
            segmentation = self._predict_raw(raw=inst, picks=picks, reject_by_annotation=reject_by_annotation,
                                                half_window_size=half_window_size,
                                                factor=factor, crit=crit,
                                                min_segment_lenght=min_segment_lenght,
                                                rejected_first_last_segments=rejected_first_last_segments,
                                                verbose=verbose)

        if isinstance(inst, BaseEpochs):
            segmentation = self._predict_epochs(epochs=inst, picks=picks,
                                                half_window_size=half_window_size,
                                                factor=factor, crit=crit,
                                                min_segment_lenght=min_segment_lenght,
                                                rejected_first_last_segments=rejected_first_last_segments,
                                                verbose=verbose)
        return(segmentation)

    def plot(self) -> matplotlib.figure.Figure:
        """Plot cluster centers as topomaps.

        Returns
        ----------
        fig :  matplotlib.figure.Figure
            The figure.
        """
        self._check_fit()
        fig, axs = plot_cluster_centers(self.cluster_centers_, self.info, self.names)
        return(fig, axs)

    def rename_clusters(self, names:list):
        """Name cluster centers. Operate in place.

        Parameters
        ----------
        names : list
            The name to give to cluster centers.

        Returns
        ----------
        self : self
            The modfied instance.
        """
        self._check_fit()
        if len(names) != self.n_clusters:
            raise ValueError(f"Names must have the same length as number of states but {len(names)} "
                             f"were given for {len(self.n_clusters)} clusters.")
        if len(set(names))!= len(names):
            raise ValueError("Can't name 2 clusters with the same name.")
        self.names = names
        return(self)

    def reorder(self, order: list):
        """Reorder cluster centers. Operate in place.

        Parameters
        ----------
        order : list
            The new cluster centers order.

        Returns
        ----------
        self : self
            The modfied instance.
        """
        self._check_fit()
        if (np.sort(order) != np.arange(0,self.n_clusters, 1)).any():
            raise ValueError('Order contains unexpected values')
           
        self.cluster_centers_ = self.cluster_centers_[order]
        self.names = [self.names[o] for o in order]
        return(self)

def _prepare_fit_raw(raw, picks, start, stop, reject_by_annotation):
    reject_by_annotation = 'omit' if reject_by_annotation else None
    start, stop = _check_start_stop(raw, start, stop)
    data = raw.get_data(picks=picks, start=start, stop=stop,
                        reject_by_annotation=reject_by_annotation)
    return(data)

def _prepare_fit_epochs(epochs, picks):
    data = epochs.get_data(picks=picks)
    data = np.swapaxes(data,0,1)
    data = data.reshape(data.shape[0], -1)
    return(data)

@fill_doc
class ModKMeans(BaseClustering):
    """Modified K-Means Clustering algorithm.

    Parameters
    ----------
    n_clusters : int
        The number of clusters to form as well as the number of centroids to generate.
    %(random_state)s
        As estimation can be non-deterministic it can be useful to fix the
        random state to have reproducible results.
    n_init : int
        Number of time the k-means algorithm will be run with different centroid seeds.
        The final result will be the run with highest global explained variance.
        Default=100
    max_iter : int
        Maximum number of iterations of the k-means algorithm for a single run.
        Default=300
    tol : float
        Relative tolerance with regards estimate residual noise in the cluster centers of two consecutive iterations to declare convergence.

    Attributes
    ----------
    n_clusters : int
        The number of clusters to form as well as the number of centroids to generate.
    current_fit : bool
        Flag informing about which data type (raw, epochs or evoked) was used for the fit.
    cluster_centers_ : :class:`numpy.ndarray`, shape ``(n_clusters, n_channels)``
        If fitted, cluster centers (i.e Microstates maps)
    info : dict
        If fitted, :class:`Measurement info <mne.Info>` of fitted instance, else None
    GEV_ : float
        If fitted, the Global explained Variance explained all clusters centers.
    """
    def __init__(self,
                 random_state=None,
                 n_init=100,
                 max_iter=300,
                 tol=1e-6,
                 *args,  **kwargs):
        super().__init__(*args, **kwargs)
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = check_random_state(random_state)
   
    def _fit_data(self, data: np.ndarray,  n_jobs: int = 1, verbose=None) -> ModKMeans:
        logger.info(f'Running Kmeans for {self.n_clusters} clusters centers with {self.n_init} random initialisations.')
        inits = self.random_state.randint(low=0, high=100*self.n_init, size=(self.n_init))
        if n_jobs == 1:
            best_gev = 0
            for init in inits:
                gev, maps, segmentation = _run_mod_kmeans(data, n_clusters=self.n_clusters,
                                                          max_iter=self.max_iter,
                                                          random_state=init,
                                                          tol=self.tol, verbose=verbose)
                if gev > best_gev:
                    best_gev, best_maps, best_segmentation = gev, maps, segmentation
        else:
            parallel, p_fun, _ = parallel_func(_run_mod_kmeans,
                                               total=self.n_init,
                                               n_jobs=n_jobs)
            runs = parallel(p_fun(data, n_clusters=self.n_clusters,
                                  max_iter=self.max_iter,
                                  random_state=init,
                                  tol=self.tol, verbose=verbose) for init in inits)
            gevs = [run[0] for run in runs]
            best_run = np.argmax(gevs)
            best_gev, best_maps, best_segmentation = runs[best_run]
            logger.info(f'Selecting run with highest GEV = {best_gev}%.')
        return(best_maps, best_gev, best_segmentation)

    @verbose
    def fit(self, inst, start=None, stop=None,
            reject_by_annotation=True,
            n_jobs=1,
            verbose=None):
        """Segment Instance into microstate sequence.

        Parameters
        ----------
        inst : :class:`mne.io.BaseRaw`, :class:`mne.Epochs`
            Instance containing data to transform to cluster-distance space (absolute spatial correlation).
        reject_by_annotation : bool
            Whether to reject by annotation. If True (default), segments annotated with description starting with ‘bad’ are omitted.
            If False,no rejection is done.
        %(n_jobs)s
        %(raw_tmin)s
        %(raw_tmax)s
        %(verbose)s
        """
        _validate_type(inst, (BaseRaw, BaseEpochs), 'inst', 'Raw or Epochs')
        reject_by_annotation = _check_reject_by_annotation(reject_by_annotation)
        n_jobs = check_n_jobs(n_jobs)

        if len(inst.info['bads']) != 0:
            warn('Bad channels are present in the recording. '
                 'They will still be used to compute microstate topographies. '
                 'Consider using instance.pick() or instance.interpolate_bads()'
                 ' before fitting.')

        inst = inst.copy()
        picks = _picks_to_idx(inst.info, self.picks,
                              exclude=[], allow_empty=False)

        if isinstance(inst, BaseRaw):
            data = _prepare_fit_raw(inst, picks, start, stop, reject_by_annotation)
            current_fit = 'Raw'

        elif isinstance(inst, BaseEpochs):
            data = _prepare_fit_epochs(inst, picks)
            current_fit = 'Epochs'

        logger.info(f'Fitting modified Kmeans with {current_fit} data')

        cluster_centers, GEV, labels =  self._fit_data(data=data, n_jobs=n_jobs, verbose=verbose)
        self.cluster_centers_ = cluster_centers
        self.current_fit = current_fit
        self.info = pick_info(inst.info, picks)
        self.GEV_ = GEV
        self.fitted_data_ = data
        self.labels_ = labels
        return()