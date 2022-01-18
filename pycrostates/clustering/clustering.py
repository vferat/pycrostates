import itertools
from typing import Tuple
from copy import deepcopy

import scipy
import numpy as np
from mne import BaseEpochs, pick_info
from mne.io import BaseRaw, RawArray
from mne.parallel import parallel_func
from mne.annotations import _annotations_starts_stops
from mne.io.pick import _picks_to_idx
from mne.preprocessing.ica import _check_start_stop

from ..viz import plot_cluster_centers
from ..segmentation import RawSegmentation, EpochsSegmentation
from ..utils import _corr_vectors
from ..utils._logs import logger, verbose
from ..utils._docs import fill_doc
from ..utils._checks import (_check_type, _check_ch_names, _check_n_jobs)


@verbose
def _compute_maps(data, n_states=4, max_iter=1000, tol=1e-6,
                  random_seed=None, verbose=None):
    """
    Comptues maps.
    Based on mne_microstates by Marijn van Vliet <w.m.vanvliet@gmail.com>
    https://github.com/wmvanvliet/mne_microstates/blob/master/microstates.py
    """
    # -- handle zeros maps --
    # zero map can be due to non data in the recording, it's unlikly that all
    # channels recorded the same value at the same time (=0 due to average
    # reference)
    data = data[:, np.linalg.norm(data.T, axis=1) != 0]
    n_channels, n_samples = data.shape
    data_sum_sq = np.sum(data ** 2)
    # Select random timepoints for our initial topographic maps
    random_state = np.random.RandomState(random_seed)
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
                # logger.info('Some microstates are never activated')
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
            # logger.info('Converged at %d iterations.', iteration)
            break

        prev_residual = residual
    else:
        logger.info('Modified K-means algorithm failed to converge.')

    return maps


@verbose
def _run_mod_kmeans(data: np.ndarray, n_clusters=4,
                    max_iter=100, random_seed=None,
                    tol=1e-6, verbose=None
                    ) -> Tuple[float, np.ndarray, np.ndarray]:
    gfp_sum_sq = np.sum(data ** 2)
    maps = _compute_maps(data, n_clusters, max_iter=max_iter,
                         random_seed=random_seed, tol=tol, verbose=verbose)
    activation = maps.dot(data)
    segmentation = np.argmax(np.abs(activation), axis=0)
    map_corr = _corr_vectors(data, maps[segmentation].T)
    # Compare across iterations using global explained variance (GEV)
    gev = np.sum((data * map_corr) ** 2) / gfp_sum_sq
    return gev, maps, segmentation


def _segment(data, states, half_window_size=3, factor=0, crit=10e-6):
    data = data - np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    data_std[data_std == 0] = 1  # std == 0 -> null map
    data = data / data_std

    states = states.T
    states = states - np.mean(states, axis=0)
    states = states / np.std(states, axis=0)
    states = states.T

    Ne, Nt = data.shape
    Nu = states.shape[0]
    Vvar = np.sum(data * data, axis=0)
    rmat = np.tile(np.arange(0, Nu), (Nt, 1)).T

    labels_all = np.argmax(np.abs(np.dot(states, data)), axis=0)

    # TODO: Check parenthesis for each Vvar; doesn't look consistent.

    if factor != 0:
        w = np.zeros((Nu, Nt))
        w[(rmat == labels_all)] = 1
        e = np.sum(
            Vvar - np.sum(np.dot(w.T, states).T * data, axis=0) ** 2 /
            (Nt * (Ne - 1))
            )

        window = np.ones((1, 2*half_window_size+1))

        S0 = 0
        while True:
            Nb = scipy.signal.convolve2d(w, window, mode='same')
            x = (np.tile(Vvar, (Nu, 1)) - (np.dot(states, data))**2) / \
                (2 * e * (Ne - 1)) - factor * Nb
            dlt = np.argmin(x, axis=0)

            labels_all = dlt
            w = np.zeros((Nu, Nt))
            w[(rmat == labels_all)] = 1
            Su = np.sum(
                Vvar - np.sum(np.dot(w.T, states).T * data, axis=0) ** 2) / \
                (Nt * (Ne - 1))
            if np.abs(Su - S0) <= np.abs(crit * Su):
                break
            S0 = Su

    labels = labels_all + 1
    return labels


def _rejected_first_last_segments(segmentation):
    # set first segment to unlabeled
    i = 0
    first_label = segmentation[i]
    if first_label != 0:
        while segmentation[i] == first_label and i < len(segmentation) - 1:
            segmentation[i] = 0
            i += 1
    # set last segment to unlabeled
    i = len(segmentation) - 1
    last_label = segmentation[i]
    if last_label != 0:
        while segmentation[i] == last_label and i > 0:
            segmentation[i] = 0
            i -= 1
    return segmentation


def _reject_small_segments(segmentation, data, min_segment_length):
    new_segmentation = segmentation.copy()
    small_segment = True
    while small_segment:
        segments = [list(group) for _,
                    group in itertools.groupby(new_segmentation)]
        current_idx = 0
        small_segment = False
        for i, segment in enumerate(segments):
            if i not in [0, len(segments)-1]:
                if len(segment) < min_segment_length and segment[0] != 0:
                    small_segment = True
                    left_idx = current_idx
                    right_idx = current_idx + len(segment)
                    new_segment = new_segmentation[left_idx:right_idx]

                    while len(new_segment) != 0:
                        left_corr = np.abs(_corr_vectors(
                            data[:, left_idx - 1].T, data[:, left_idx].T,))
                        right_corr = np.abs(_corr_vectors(
                            data[:, right_idx].T, data[:, right_idx - 1].T))

                        if left_corr < right_corr:
                            new_segmentation[right_idx - 1] = \
                                new_segmentation[right_idx]
                            right_idx -= 1
                        else:
                            new_segmentation[left_idx] = \
                                new_segmentation[left_idx - 1]
                            left_idx += 1
                        new_segment = new_segmentation[left_idx:right_idx]
                    break
            current_idx += len(segment)
    return new_segmentation


class BaseClustering:
    """Base Class for Microstate Clustering algorithms.

    Parameters
    ----------
    n_clusters : int
        The number of clusters to form as well as the number of centroids to
        generate.
    picks : str | list | slice | None
        Channels to include. Slices and lists of integers will be interpreted
        as channel indices. In lists, channel type strings (e.g.,
        ['meg', 'eeg']) will pick channels of those types, channel name strings
        (e.g., ['MEG0111', 'MEG2623'] will pick the given channels.
        Can also be the string values “all” to pick all channels, or “data” to
        pick data channels. None will pick all channels.
        Note that channels in info['bads'] will be included if their names or
        indices are explicitly provided. Default to 'eeg'.

    Attributes
    ----------
    n_clusters : int
        The number of clusters to form as well as the number of centroids to
        generate.
    current_fit : bool
        Flag informing about which data type (raw, epochs or evoked) was used
        for the fit.
    cluster_centers_ : `~numpy.array`, shape ``(n_clusters, n_channels)``
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
        s = '%s | n = %s cluster centers | %s'
        return s % (self.__class__.__name__, self.n_clusters, self.current_fit)

    def copy(self):
        """
        Return a copy of the instance.
        """
        return deepcopy(self)

    def _check_fit(self):
        if self.current_fit == 'unfitted':
            raise ValueError(
                'Algorithm must be fitted before using '
                f'{self.__class__.__name__}')

    def get_cluster_centers(self):
        """
        Get cluster centers as a `~numpy.array`.
        """
        self._check_fit()
        cluster_centers = self.cluster_centers_.copy()
        return cluster_centers

    def get_cluster_centers_as_raw(self):
        """
        Get cluster centers as a :class:`~mne.io.Raw`
        """
        self._check_fit()
        cluster_centers_raw = RawArray(self.cluster_centers_.T, self.info)
        return cluster_centers_raw

    def invert_polarity(self, invert):
        """Invert map polarities.

        This method is for visualisation purpose only and has no impact on
        further processing as map polarities are ignored. Operates in-place.

        Parameters
        ----------
        invert : list of bool
            List of bool of length ``n_clusters``.
            True will invert map polarity, while False will have no effect.
        """
        self._check_fit()
        cluster_centers = self.cluster_centers_
        for c, cluster in enumerate(cluster_centers):
            if invert[c]:
                cluster_centers[c] = - cluster
        self.cluster_centers_ = cluster_centers

    @verbose
    def _predict_raw(self, raw, picks, reject_by_annotation, half_window_size,
                     factor, crit, min_segment_length,
                     rejected_first_last_segments, verbose=None):
        data = raw.get_data(picks=picks)
        if not reject_by_annotation:
            segmentation = _segment(data, self.cluster_centers_,
                                    half_window_size, factor, crit)
            if rejected_first_last_segments:
                segmentation = _rejected_first_last_segments(segmentation)

        else:
            onsets, _ends = _annotations_starts_stops(raw, ['BAD'])
            if len(onsets) == 0:
                segmentation = _segment(data,
                                        self.cluster_centers_,
                                        half_window_size, factor,
                                        crit)
                if rejected_first_last_segments:
                    segmentation = _rejected_first_last_segments(segmentation)

            else:
                onsets = onsets.tolist()
                onsets.append(data.shape[-1] - 1)
                _ends = _ends.tolist()
                ends = [0]
                ends.extend(_ends)

                segmentation = np.zeros(data.shape[-1])
                for onset, end in zip(onsets, ends):
                    # small segments can't be smoothed
                    if onset - end >= 2 * half_window_size + 1:
                        sample = data[:, end:onset]
                        labels = _segment(sample, self.cluster_centers_,
                                          half_window_size, factor, crit)
                        if rejected_first_last_segments:
                            labels = _rejected_first_last_segments(labels)
                        segmentation[end:onset] = labels

        if min_segment_length > 0:
            segmentation = _reject_small_segments(
                segmentation, data, min_segment_length)

        segmentation = RawSegmentation(segmentation=segmentation,
                                       inst=raw,
                                       cluster_centers=self.cluster_centers_,
                                       names=self.names)
        return segmentation

    @verbose
    def _predict_epochs(self, epochs, picks, half_window_size, factor, crit,
                        min_segment_length, rejected_first_last_segments,
                        verbose=None):
        data = epochs.get_data(picks=picks)
        segments = list()
        for epoch in data:
            segment = _segment(epoch, self.cluster_centers_,
                               half_window_size, factor, crit)

            if min_segment_length > 0:
                segment = _reject_small_segments(segment, epoch,
                                                 min_segment_length)

            if rejected_first_last_segments:
                segment = _rejected_first_last_segments(segment)
            segments.append(segment)

        segments = np.array(segments)
        segmentation = EpochsSegmentation(
            segmentation=segments, inst=epochs,
            cluster_centers=self.cluster_centers_, names=self.names)
        return segmentation

    @fill_doc
    @verbose
    def predict(self,  inst,
                reject_by_annotation=True,
                factor=0,
                half_window_size=3,
                crit=10e-6,
                min_segment_length=0,
                rejected_first_last_segments=True,
                verbose=None):
        """Predict Microstates labels using competitive fitting.

        Parameters
        ----------
        inst : `~mne.io.Raw`, `~mne.Epochs`
            Instance containing data to predict.
        factor: int
            Factor used for label smoothing. 0 means no smoothing.
            Defaults to 0.
        half_window_size: int
            Number of samples used for the half windows size while smoothing
            labels.
            Has no effect if factor = 0.
            Window size = 2 * half_window_size + 1
        crit: float
            Converge criterion. Default to 10e-6.
        min_segment_length: int
            Minimum segment length (in samples). If a segment is shorter than
            this value, it will be recursively reassigned to neighboring
            segments based on absolute spatial correlation.
        rejected_first_last_segments: bool
            If True, set first and last segments to unlabeled.
            Default to True.
        reject_by_annotation : bool
            Whether to reject by annotation. If True (default), segments
            annotated with description starting with `bad` are omitted.
            If False, no rejection is done.
        %(verbose)s

        Returns
        ----------
        segmentation : `~numpy.ndarray`
            Microstate sequence derivated from Instance data. Timepoints are
            labeled according to cluster centers number: 1 for the first
            center, 2 for the second ect.. 0 is used for unlabeled time points.
        """
        self._check_fit()
        _check_type(inst, (BaseRaw, BaseEpochs))
        inst = inst.copy()
        picks = _picks_to_idx(inst.info, self.picks,
                              exclude=[], allow_empty=False)
        inst = inst.pick(picks)

        _check_ch_names(self, inst,
                        inst1_name=type(self).__name__,
                        inst2_name='inst')
        if factor == 0:
            logger.info('Segmenting data without smoothing')
        if factor != 0:
            logger.info(
                'Segmenting data with factor %s and effective smoothing '
                'window size: %s (ms)',
                factor, (2*half_window_size+1) / inst.info["sfreq"] * 1000)
        if min_segment_length > 0:
            logger.info('Rejecting segments shorter than %s (ms)',
                        min_segment_length / inst.info["sfreq"] * 1000)
        if rejected_first_last_segments:
            logger.info('Rejecting first and last segment')

        if isinstance(inst, BaseRaw):
            segmentation = self._predict_raw(
                raw=inst,
                picks=picks,
                reject_by_annotation=reject_by_annotation,
                half_window_size=half_window_size,
                factor=factor, crit=crit,
                min_segment_length=min_segment_length,
                rejected_first_last_segments=rejected_first_last_segments,
                verbose=verbose)

        if isinstance(inst, BaseEpochs):
            segmentation = self._predict_epochs(
                epochs=inst,
                picks=picks,
                half_window_size=half_window_size,
                factor=factor, crit=crit,
                min_segment_length=min_segment_length,
                rejected_first_last_segments=rejected_first_last_segments,
                verbose=verbose)
        return segmentation

    def plot(self):
        """Plot cluster centers as topomaps.

        Returns
        ----------
        fig : matplotlib.figure.Figure
            The figure.
        """
        self._check_fit()
        fig, axs = plot_cluster_centers(self.cluster_centers_, self.info,
                                        self.names)
        return fig, axs

    def rename_clusters(self, names: list):
        """
        Name cluster centers. Operate in-place.

        Parameters
        ----------
        names : list
            The name to give to cluster centers.

        Returns
        ----------
        self : self
            The modfied instance.
        """
        # TODO: should use a similar system to mapping types/names on ch in MNE
        # with an argument `mapping` of type dict.
        self._check_fit()
        if len(names) != self.n_clusters:
            raise ValueError(
                "Names must have the same length as number of states but "
                f"{len(names)} were given for {len(self.n_clusters)} clusters."
                )
        if len(set(names)) != len(names):
            raise ValueError("Can't name 2 clusters with the same name.")
        self.names = names

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
        if len(order) != self.n_clusters:
            raise ValueError('Order contains unexpected values')
               
        if (np.sort(order) != np.arange(0, self.n_clusters, 1)).any():
            raise ValueError('Order contains unexpected values')

        self.cluster_centers_ = self.cluster_centers_[order]
        self.names = [self.names[o] for o in order]


# TODO: could be merge into a single `_prepare` function/method thus removing
# the check above on BaseRaw/BaseEpochs and the duplication for `segmentation`
def _prepare_fit_raw(raw, picks, start, stop, reject_by_annotation):
    reject_by_annotation = 'omit' if reject_by_annotation else None
    start, stop = _check_start_stop(raw, start, stop)
    data = raw.get_data(picks=picks, start=start, stop=stop,
                        reject_by_annotation=reject_by_annotation)
    return data


def _prepare_fit_epochs(epochs, picks):
    data = epochs.get_data(picks=picks)
    data = np.swapaxes(data, 0, 1)
    data = data.reshape(data.shape[0], -1)
    return data


@fill_doc
class ModKMeans(BaseClustering):
    """Modified K-Means Clustering algorithm.

    Parameters
    ----------
    n_clusters : int
        The number of clusters to form as well as the number of centroids to
        generate.
    random_seed : float
        As estimation can be non-deterministic it can be useful to fix the
        random state to have reproducible results.
    n_init : int
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final result will be the run with highest global
        explained variance. Default=100
    max_iter : int
        Maximum number of iterations of the k-means algorithm for a single run.
        Default=300
    tol : float
        Relative tolerance with regards estimate residual noise in the cluster
        centers of two consecutive iterations to declare convergence.

    Attributes
    ----------
    n_clusters : int
        The number of clusters to form as well as the number of centroids to
        generate.
    current_fit : bool
        Flag informing about which data type (raw, epochs or evoked) was used
        for the fit.
    cluster_centers_ : `~numpy.array`, shape ``(n_clusters, n_channels)``
        If fitted, cluster centers (i.e Microstates maps)
    info : dict
        If fitted, :class:`Measurement info <mne.Info>` of fitted instance,
        else None
    GEV_ : float
        If fitted, the Global explained Variance explained all clusters
        centers.
    """
    def __init__(self,
                 random_seed=None,
                 n_init=100,
                 max_iter=300,
                 tol=1e-6,
                 *args,  **kwargs):
        super().__init__(*args, **kwargs)
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_seed = random_seed

    def _fit_data(self, data: np.ndarray,  n_jobs: int = 1, verbose=None):
        logger.info('Running Kmeans for %s clusters centers with %s random '
                    'initialisations.', self.n_clusters, self.n_init)
        random_state = np.random.RandomState(self.random_seed)
        inits = random_state.randint(
            low=0, high=100*self.n_init, size=(self.n_init))
        if n_jobs == 1:
            best_gev = 0
            for init in inits:
                gev, maps, segmentation = _run_mod_kmeans(
                    data, n_clusters=self.n_clusters,
                    max_iter=self.max_iter, random_seed=init,
                    tol=self.tol, verbose=verbose)
                if gev > best_gev:
                    best_gev, best_maps, best_segmentation = \
                        gev, maps, segmentation
        else:
            parallel, p_fun, _ = parallel_func(_run_mod_kmeans,
                                               total=self.n_init,
                                               n_jobs=n_jobs)
            runs = parallel(
                p_fun(data, n_clusters=self.n_clusters, max_iter=self.max_iter,
                      random_seed=init, tol=self.tol, verbose=verbose)
                for init in inits)
            gevs = [run[0] for run in runs]
            best_run = np.argmax(gevs)
            best_gev, best_maps, best_segmentation = runs[best_run]
            logger.info('Selecting run with highest GEV = %.2f%%.', best_gev)
        return best_maps, best_gev, best_segmentation

    @verbose
    def fit(self, inst, start=None, stop=None, reject_by_annotation=True,
            n_jobs=1, verbose=None):
        """Segment Instance into microstate sequence.

        Parameters
        ----------
        inst : `~mne.io.Raw`, `~mne.Epochs`
            Instance containing data to transform to cluster-distance space
            (absolute spatial correlation).
        reject_by_annotation : bool
            Whether to reject by annotation. If True (default), segments
            annotated with description starting with ‘bad’ are omitted.
            If False, no rejection is done.
        %(n_jobs)s
        %(raw_tmin)s
        %(raw_tmax)s
        %(verbose)s
        """
        _check_type(inst, (BaseRaw, BaseEpochs))
        n_jobs = _check_n_jobs(n_jobs)

        if len(inst.info['bads']) != 0:
            logger.warning(
                'Bad channels are present in the recording. They will still '
                'be used to compute microstate topographies. Consider using '
                'instance.pick() or instance.interpolate_bads() before '
                'fitting.')

        inst = inst.copy()
        picks = _picks_to_idx(inst.info, self.picks,
                              exclude=[], allow_empty=False)

        if isinstance(inst, BaseRaw):
            data = _prepare_fit_raw(inst, picks, start, stop,
                                    reject_by_annotation)
            current_fit = 'Raw'

        elif isinstance(inst, BaseEpochs):
            data = _prepare_fit_epochs(inst, picks)
            current_fit = 'Epochs'

        logger.info('Fitting modified Kmeans with %s data', current_fit)

        cluster_centers, GEV, labels = self._fit_data(data, n_jobs, verbose)
        self.cluster_centers_ = cluster_centers
        self.current_fit = current_fit
        self.info = pick_info(inst.info, picks)
        self.GEV_ = GEV
        self.fitted_data_ = data
        self.labels_ = labels
