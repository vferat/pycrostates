from abc import ABC, abstractmethod
from copy import copy, deepcopy
from itertools import groupby
from typing import Union

from mne import BaseEpochs, pick_info
from mne.annotations import _annotations_starts_stops
from mne.io import BaseRaw, Info, RawArray
from mne.io.pick import _picks_to_idx
import numpy as np
from scipy.signal import convolve2d

from ..segmentation import RawSegmentation, EpochsSegmentation
from ..utils import _corr_vectors, _compare_infos
from ..utils._checks import _check_type, _check_value, _check_n_jobs
from ..utils._docs import fill_doc
from ..utils._logs import verbose, logger
from ..viz import plot_cluster_centers


class _BaseCluster(ABC):
    """
    Base Class for Microstates Clustering algorithms.
    """

    @abstractmethod
    def __init__(self):
        self._n_clusters = None
        self._clusters_names = None
        self._cluster_centers_ = None

        # fit variables
        self._picks = None
        self._info = None
        self._fitted_data = None
        self._labels_ = None
        self._fitted = False

    def __repr__(self) -> str:
        name = self.__class__.__name__
        if self.fitted:
            s = f'<{name} | fitted on n = {self.n_clusters} cluster centers>'
        else:
            s = f'<{name} | not fitted>'
        return s

    def copy(self, deep=True):
        """Returns a copy of the instance.

        Parameters
        ----------
        deep : bool
            If True, `~copy.deepcopy` is used instead of `~copy.copy`.
        """
        if deep:
            return deepcopy(self)
        else:
            return copy(self)

    def _check_fit(self):
        """Check if the cluster is fitted."""
        if not self.fitted:
            raise RuntimeError(
                'Clustering algorithm must be fitted before using '
                f'{self.__class__.__name__}')
        # sanity-check
        assert self.cluster_centers_ is not None
        assert self.picks is not None
        assert self.info is not None
        assert self.fitted_data is not None
        assert self.labels_ is not None

    @abstractmethod
    @fill_doc
    def fit(self, inst, picks='eeg', tmin=None, tmax=None,
            reject_by_annotation=True, n_jobs=1):
        """
        Segment `~mne.io.Raw` or `~mne.Epochs` instance into microstate
        sequence.

        Parameters
        ----------
        %(fit_inst)s
        %(picks_all)s
        %(tmin_raw)s
        %(tmax_raw)s
        %(reject_by_annotation_raw)s
        %(n_jobs)s
        """
        # TODO: Maybe those parameters should be moved here instead of docdict?
        _check_type(inst, (BaseRaw, BaseEpochs), item_name='inst')
        _check_type(tmin, (None, 'numeric'), item_name='tmin')
        _check_type(tmax, (None, 'numeric'), item_name='tmax')
        reject_by_annotation = _BaseCluster._check_reject_by_annotation(
            reject_by_annotation)
        n_jobs = _check_n_jobs(n_jobs)

        # picks
        bads_inc = _picks_to_idx(inst.info, picks, none='all', exclude=[])
        picks = _picks_to_idx(inst.info, picks, none='all', exclude='bads')
        diff = set(bads_inc) - set(picks)
        if len(diff) != 0:
            if len(diff) == 1:
                msg = "Channel %s is set as bad and ignored. To include " + \
                      "it, either remove it from 'inst.info['bads'] or " + \
                      "provide its name explicitly in the 'picks' argument."
            else:
                msg = "Channels %s are set as bads and ignored. To " + \
                      "include them, either remove them from " + \
                      "'inst.info['bads'] or provide their names " + \
                      "explicitly in the 'picks' argument."
            logger.warning(
                msg, ', '.join(inst.info['ch_names'][k] for k in diff))
            del msg
        del bads_inc
        del diff

        # tmin/tmax
        # check positiveness
        for name, arg in (('tmin', tmin), ('tmax', tmax)):
            if arg is None:
                continue
            if arg < 0:
                raise ValueError(
                    f"Argument '{name}' must be positive. Provided '{arg}'.")
        # check tmax is shorter than raw
        if tmax is not None and inst.times[-1] < tmax:
            raise ValueError(
                "Argument 'tmax' must be shorter than the instance length. "
                f"Provided: '{tmax}', larger than {inst.times[-1]}s instance.")
        # check that tmax is larger than tmin
        if tmax is not None and tmin is not None and tmax <= tmin:
            raise ValueError(
                "Argument 'tmax' must be strictly larger than 'tmin'. "
                f"Provided 'tmin' -> '{tmin}' and 'tmax' -> '{tmax}'.")
        elif tmin is not None and inst.times[-1] <= tmin:
            raise ValueError(
                "Argument 'tmin' must be shorter than the instance length. "
                f"Provided: '{tmin}', larger than {inst.times[-1]}s instance.")

        # retrieve numpy array
        kwargs = dict() if isinstance(inst, BaseEpochs) \
            else dict(reject_by_annotation=reject_by_annotation)
        data = inst.get_data(picks=picks, tmin=tmin, tmax=tmax, **kwargs)
        # reshape if inst is Epochs
        if isinstance(inst, BaseEpochs):
            data = np.swapaxes(data, 0, 1)
            data = data.reshape(data.shape[0], -1)

        # store picks and info
        self._picks = picks
        self._info = pick_info(inst.info, picks)
        self._fitted_data = data

        return data

    def rename_clusters(self, mapping: dict = None,
                        new_names: Union[list, tuple] = None):
        """
        Rename the clusters in-place.

        Parameters
        ----------
        mapping : dict
            Mapping from the old names to the new names.
            key: old name, value: new name.
        new_names : list | tuple
            1D iterable containing the new cluster names.
        """
        self._check_fit()

        if mapping is not None and new_names is not None:
            raise ValueError(
                "Only one of 'mapping' or 'new_names' must be provided.")

        elif mapping is not None:
            _check_type(mapping, (dict, ), item_name='mapping')
            for key in mapping:
                _check_value(key, self.clusters_names, item_name='old name')
            for value in mapping.values():
                _check_type(value, (str, ), item_name='new name')

        elif new_names is not None:
            _check_type(new_names, (list, tuple), item_name='new_names')
            if len(new_names) != self._n_clusters:
                raise ValueError(
                    "Argument 'new_names' should contain 'n_clusters': "
                    f"{self._n_clusters} elements. "
                    f"Provided '{len(new_names)}'.")

            # sanity-check
            assert len(self._clusters_names) == len(new_names)

            # convert to dict
            mapping = {old_name: new_names[k]
                       for k, old_name in enumerate(self._clusters_names)}

        else:
            logger.warning(
                "Either 'mapping' or 'new_names' should not be 'None' "
                "for method 'rename_clusters' to operate.")
            return

        self._clusters_names = [mapping[name] if name in mapping else name
                                for name in self._clusters_names]

    def reorder_clusters(self, mapping: dict = None,
                         order: Union[list, tuple] = None):
        """
        Reorder the clusters in-place. The positions are 0-indexed.

        Parameters
        ----------
        mapping : dict
            Mapping from the old order to the new order.
            key: old position, value: new position.
        order : list | tuple
            1D iterable containing the new order.
        """
        self._check_fit()

        if mapping is not None and order is not None:
            raise ValueError(
                "Only one of 'mapping' or 'order' must be provided.")

        elif mapping is not None:
            _check_type(mapping, (dict, ), item_name='mapping')
            valids = tuple(range(self._n_clusters))
            for key in mapping:
                _check_value(key, valids, item_name='old position')
            for value in mapping.values():
                _check_value(value, valids, item_name='new position')

            inverse_mapping = {value: key for key, value in mapping.items()}

            # check uniqueness
            if len(set(mapping.values())) != len(mapping.values()):
                raise ValueError(
                    'Position in the new order can not be repeated.')
            # check that a cluster is not moved twice
            for key in mapping:
                if key in mapping.values():
                    raise ValueError(
                        "A position can not be present in both the old and "
                        f"new order. Position '{key}' is mapped to "
                        f"'{mapping[key]}' and position "
                        f"'{inverse_mapping[key]}' is mapped to '{key}'.")

            # convert to list
            order = list(range(self._n_clusters))
            for key, value in mapping.items():
                order[key] = value
                order[value] = key
            # sanity-check
            assert len(set(order)) == self._n_clusters

        elif order is not None:
            _check_type(order, (list, tuple, np.ndarray), item_name='order')
            if isinstance(order, np.ndarray) and len(order.shape) != 1:
                raise ValueError(
                    "Argument 'order' should be a 1D iterable and not a "
                    f"{len(order.shape)}D iterable.")
            valids = tuple(range(self._n_clusters))
            for elt in order:
                _check_value(elt, valids, item_name='order')
            if len(order) != self._n_clusters:
                raise ValueError(
                    "Argument 'order' should contain 'n_clusters': "
                    f"{self._n_clusters} elements. Provided '{len(order)}'.")
            order = list(order)

        else:
            logger.warning("Either 'mapping' or 'order' should not be 'None' "
                           "for method 'reorder_clusters' to operate.")
            return

        # re-order
        self._cluster_centers_ = self._cluster_centers_[order]
        self._clusters_names = [self._clusters_names[k] for k in order]
        new_labels = np.full(self._labels_.shape, -1)
        for k in range(0, self.n_clusters):
            new_labels[self._labels_ == k] = order[k]
        self._labels_ = new_labels

    def invert_polarity(self, invert):
        """
        Invert map polarities for vizualisation purposes. Operates in-place.

        Parameters
        ----------
        invert : bool | list of bool
            List of bool of length ``n_clusters``.
            True will invert map polarity, while False will have no effect.
            If a bool is provided, it is applied to all maps.
        """
        self._check_fit()

        # Check argument
        invert = _check_type(invert, (bool, list, tuple, np.ndarray),
                             item_name='invert')
        if isinstance(invert, bool):
            invert = [invert] * self._n_clusters
        elif isinstance(invert, (list, tuple)):
            for inv in invert:
                _check_type(inv, (bool, ), item_name='invert')
        elif isinstance(invert, np.ndarray):
            if len(invert.shape) != 1:
                raise ValueError(
                    "Argument 'invert' should be a 1D iterable and not a "
                    f"{len(invert.shape)}D iterable.")
            for inv in invert:
                _check_type(inv, (bool, np.bool_), item_name='invert')
        if len(invert) != self._n_clusters:
            raise ValueError(
                "Argument 'invert' should be either a bool or a list of bools "
                f"of length 'n_clusters' ({self._n_clusters}). The provided "
                f"'invert' length is '{len(invert)}'.")

        # Invert maps
        for k, cluster in enumerate(self._cluster_centers_):
            if invert[k]:
                self._cluster_centers_[k] = - cluster

    def plot(self, block=False):
        """
        Plot cluster centers as topographic maps.

        Returns
        -------
        fig : :class:`matplotlib.figure.Figure`
            Figure
        ax : :class:`matplotlib.axes.Axes`
            Axis
        """
        self._check_fit()
        return plot_cluster_centers(self._cluster_centers_, self._info,
                                    self._clusters_names, block)

    @verbose
    def predict(self, inst, factor=0, half_window_size=3, tol=10e-6,
                min_segment_length=0, reject_edges=True,
                reject_by_annotation=True, *, verbose=None):
        """Segment `~mne.io.Raw` or `~mne.Epochs` instance into microstate
        sequence.

        Parameters
        ----------
        %(predict_inst)s
        factor : int
            Factor used for label smoothing. ``0`` means no smoothing.
        half_window_size : int
            Number of samples used for the half window size while smoothing
            labels. Has no ffect if ``factor=0``. The half window size is
            defined as ``window_size = 2 * half_window_size + 1``.
        tol : float
            Convergence tolerance.
        min_segment_length : int
            Minimum segment length (in samples). If a segment is shorter than
            this value, it will be recursively reasigned to neighbouring
            segments based on absolute spatial correlation.
        reject_edges : bool
            If True, set first and last segments to unlabeled.
        %(reject_by_annotation_raw)s
        %(verbose)s

        Returns
        -------
        segmentation `RawSegmentation` | `EpochsSegmentation`
            Microstate sequence derivated from istance data. Timepoints are
            labeled according to cluster centers number: 1 for the first
            center, 2 for the second, etc..
            0 is used for unlabeled time points.
        """
        # TODO: reject_by_annotation_raw doc probably doesn't match the correct
        # argument types.
        self._check_fit()
        _check_type(inst, (BaseRaw, BaseEpochs), item_name='inst')
        _check_type(factor, ('int', ), item_name='factor')
        _check_type(half_window_size, ('int', ), item_name='half_window_size')
        _check_type(tol, ('numeric', ), item_name='tol')
        _check_type(min_segment_length, ('int', ),
                    item_name='min_segment_length')
        _check_type(reject_edges, (bool, ), item_name='reject_edges')
        _check_type(reject_by_annotation, (bool, str, None),
                    item_name='reject_by_annotation')
        if isinstance(reject_by_annotation, str):
            if reject_by_annotation == 'omit':
                reject_by_annotation = True
            else:
                logger.warning(
                    "'reject_by_annotation' can be set to 'True', 'False' or "
                    "'omit' (True). '%s' is not supported. Setting to "
                    "'False'.", reject_by_annotation)
                reject_by_annotation = False
        elif reject_by_annotation is None:
            reject_by_annotation = False

        # check that the channels match
        msg = 'Instance to segment into microstate sequence does not have ' + \
              'the same channels as the instance used for fitting.'
        try:
            info = pick_info(inst.info, self._picks)
        except IndexError:
            raise ValueError(msg)
        _compare_infos(self.info, info)

        # logging messages
        if factor == 0:
            logger.info('Segmenting data without smoothing.')
        else:
            logger.info(
                'Segmenting data with factor %s and effective smoothing '
                'window size: %.4f (ms).', factor,
                (2*half_window_size+1) / inst.info["sfreq"])
        if min_segment_length > 0:
            logger.info('Rejecting segments shorter than %.4f (ms).',
                        min_segment_length / inst.info["sfreq"])
        if reject_edges:
            logger.info('Rejecting first and last segments.')

        if isinstance(inst, BaseRaw):
            segmentation = self._predict_raw(
                inst, self._picks, factor, tol, half_window_size,
                min_segment_length, reject_edges, reject_by_annotation)
        elif isinstance(inst, BaseEpochs):
            segmentation = self._predict_epochs(
                inst, self._picks, factor, tol, half_window_size,
                min_segment_length, reject_edges)
        return segmentation

    def _predict_raw(self, raw, picks, factor, tol, half_window_size,
                     min_segment_length, reject_edges, reject_by_annotation):
        """Create segmentation for raw."""
        data = raw.get_data(picks=picks)

        if reject_by_annotation:
            # retrieve onsets/ends for BAD annotations
            onsets, ends = _annotations_starts_stops(raw, ['BAD'])
            onsets = onsets.tolist() + [data.shape[-1] - 1]
            ends = [0] + ends.tolist()

            segmentation = np.full(data.shape[-1], -1)

            for onset, end in zip(onsets, ends):
                # small segments can't be smoothed
                if factor != 0 and onset - end < 2 * half_window_size + 1:
                    continue

                data_ = data[:, end:onset]
                segment = _BaseCluster._segment(
                    data_, deepcopy(self._cluster_centers_), factor, tol,
                    half_window_size)
                if reject_edges:
                    segment = _BaseCluster._reject_edge_segments(segment)
                segmentation[end:onset] = segment

        else:
            segmentation = _BaseCluster._segment(
                data, deepcopy(self._cluster_centers_), factor, tol,
                half_window_size)
            if reject_edges:
                segmentation = _BaseCluster._reject_edge_segments(segmentation)

        if 0 < min_segment_length:
            segmentation = _BaseCluster._reject_short_segments(
                segmentation, data, min_segment_length)

        return RawSegmentation(labels=segmentation,
                               inst=raw, picks=picks,
                               cluster_centers_=self._cluster_centers_,
                               clusters_names=self._clusters_names)

    def _predict_epochs(self, epochs, picks, factor, tol, half_window_size,
                        min_segment_length, reject_edges):
        """Create segmentation for epochs."""
        data = epochs.get_data(picks=picks)
        segments = list()
        for epoch_data in data:
            segment = _BaseCluster._segment(
                epoch_data, deepcopy(self._cluster_centers_),
                factor, tol, half_window_size)

            if 0 < min_segment_length:
                segment = _BaseCluster._reject_short_segments(
                    segment, epoch_data, min_segment_length)
            if reject_edges:
                segment = _BaseCluster._reject_edge_segments(segment)

            segments.append(segment)

        return EpochsSegmentation(labels=np.array(segments),
                                  inst=epochs, picks=picks,
                                  cluster_centers_=self._cluster_centers_,
                                  clusters_names=self._clusters_names)

    # --------------------------------------------------------------------
    @staticmethod
    def _segment(data, states, factor, tol, half_window_size):
        """Create segmentation. Must operate on a copy of states."""
        data -= np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1  # std == 0 -> null map
        data /= std

        states -= np.mean(states, axis=1)[:, np.newaxis]
        states /= np.std(states, axis=1)[:, np.newaxis]

        labels = np.argmax(np.abs(np.dot(states, data)), axis=0)

        if factor != 0:
            labels = _BaseCluster._smooth_segmentation(
                data, states, labels, factor, tol, half_window_size)

        return labels

    @staticmethod
    def _smooth_segmentation(data, states, labels, factor, tol,
                             half_window_size):
        """Apply smooting. Adapted from [1].

        References
        ----------
        .. [1] R. D. Pascual-Marqui, C. M. Michel and D. Lehmann.
            Segmentation of brain electrical activity into microstates:
            model estimation and validation.
            IEEE Transactions on Biomedical Engineering,
            vol. 42, no. 7, pp. 658-665, July 1995,
            https://doi.org/10.1109/10.391164."""
        Ne, Nt = data.shape
        Nu = states.shape[0]
        Vvar = np.sum(data * data, axis=0)
        rmat = np.tile(np.arange(0, Nu), (Nt, 1)).T

        w = np.zeros((Nu, Nt))
        w[(rmat == labels)] = 1
        e = np.sum(
            Vvar - np.sum(np.dot(w.T, states).T * data, axis=0) ** 2) / \
            (Nt * (Ne - 1))
        window = np.ones((1, 2 * half_window_size + 1))

        S0 = 0
        while True:
            Nb = convolve2d(w, window, mode='same')
            x = (np.tile(Vvar, (Nu, 1)) - (np.dot(states, data)) ** 2) / \
                (2 * e * (Ne - 1)) - factor * Nb
            dlt = np.argmin(x, axis=0)

            labels = dlt
            w = np.zeros((Nu, Nt))
            w[(rmat == labels)] = 1
            Su = np.sum(
                Vvar - np.sum(np.dot(w.T, states).T * data, axis=0) ** 2) / \
                (Nt * (Ne - 1))
            if np.abs(Su - S0) <= np.abs(tol * Su):
                break
            S0 = Su

        return labels

    @staticmethod
    def _reject_short_segments(segmentation, data, min_segment_length):
        """Reject segments that are too short by replacing the labels with the
        adjacent labels based on data correlation."""
        while True:
            # list all segments
            segments = [list(group) for _, group in groupby(segmentation)]
            idx = 0  # where does the segment start

            for k, segment in enumerate(segments):
                skip_condition = [
                    k in (0, len(segments)-1),  # ignore edge segments
                    segment[0] == -1,  # ignore segments labelled with 0
                    min_segment_length <= len(segment)  # ignore large segments
                    ]
                if any(skip_condition):
                    idx += len(segment)
                    continue

                left = idx
                right = idx + len(segment) - 1
                new_segment = segmentation[left:right+1]

                while len(new_segment) != 0:
                    # compute correlation left/right side
                    left_corr = np.abs(_corr_vectors(
                        data[:, left-1].T, data[:, left].T,))
                    right_corr = np.abs(_corr_vectors(
                        data[:, right].T, data[:, right+1].T))

                    if np.abs(right_corr - left_corr) <= 1e-8:
                        # equal corr, try to do both sides
                        if len(new_segment) == 1:
                            # do only one side, left
                            segmentation[left] = segmentation[left-1]
                            left += 1
                        else:
                            # If equal, do both sides
                            segmentation[right] = segmentation[right+1]
                            segmentation[left] = segmentation[left-1]
                            right -= 1
                            left += 1
                    else:
                        if left_corr < right_corr:
                            segmentation[right] = segmentation[right+1]
                            right -= 1
                        elif right_corr < left_corr:
                            segmentation[left] = segmentation[left-1]
                            left += 1

                    # crop segment
                    new_segment = segmentation[left:right+1]

                # segments that were too short might have become long enough,
                # so list them again and check again.
                break
            else:
                break  # stop while loop because all segments are long enough

        return segmentation

    @staticmethod
    def _reject_edge_segments(segmentation):
        """Set the first and last segment as unlabeled (0)."""
        # set first segment to unlabeled
        n = (segmentation != segmentation[0]).argmax()
        segmentation[:n] = -1

        # set last segment to unlabeled
        n = np.flip((segmentation != segmentation[-1])).argmax()
        segmentation[-n:] = -1

        return segmentation

    # --------------------------------------------------------------------
    @property
    def n_clusters(self):
        """
        Number of clusters.

        :type: `int`
        """
        return self._n_clusters

    @property
    def clusters_names(self):
        """
        Name of the clusters.

        :type: `list`
        """
        return self._clusters_names

    @property
    def cluster_centers_(self):
        """
        Center of the clusters. Returns None if cluster algorithm has not been
        fitted.

        :type: `~numpy.array`
        """
        if self._cluster_centers_ is None:
            assert not self._fitted  # sanity-check
            logger.warning('Clustering algorithm has not been fitted.')
        return self._cluster_centers_

    @property
    def cluster_centers_raw(self):
        """
        Center of the clusters as a :class:`~mne.io.Raw` instance.
        Returns None if cluster algorithm has not been fitted.
        """
        if self._cluster_centers_ is None:
            assert not self._fitted  # sanity-check
            logger.warning('Clustering algorithm has not been fitted.')
            return None

        # sanity-checks
        assert self._info is not None and isinstance(self._info, Info)

        # ---------------------------------------------------------------------
        # TODO: I can't see the sfreq=-1 work well with .get_data() in the long
        # run e.g. what happens if start, stop, tmin, tmax are provided?
        # ---------------------------------------------------------------------
        info = self._info.copy()
        if hasattr(info, '_unlocked'):
            with info._unlock():
                info['sfreq'] = -1.
        else:
            info['sfreq'] = -1.

        return RawArray(self._cluster_centers_.T, info)

    @property
    def picks(self):
        """
        Picks selected when fitting the clustering algorithm.
        The picks have been converted to IDx.

        :type: `~numpy.array`
        """
        if self._picks is None:
            assert not self._fitted  # sanity-check
            logger.warning('Clustering algorithm has not been fitted.')
        return self._picks

    @property
    def info(self):
        """
        Info instance corresponding to the MNE object used to fit the
        clustering algorithm.

        :type: `mne.io.Info`
        """
        if self._info is None:
            assert not self._fitted  # sanity-check
            logger.warning('Clustering algorithm has not been fitted.')
        return self._info

    @property
    def fitted_data(self):
        """
        Data array retrieved from MNE used to fit the clustering algorithm.

        :type: `~numpy.array`
        """
        # TODO: Add shape of the numpy array to the docstring
        if self._fitted_data is None:
            assert not self._fitted  # sanity-check
            logger.warning('Clustering algorithm has not been fitted.')
        return self._fitted_data

    @property
    def labels_(self):
        """
        labels fit variable.
        """
        if self._labels_ is None:
            assert not self._fitted  # sanity-check
            logger.warning('Clustering algorithm has not been fitted.')
        return self._labels_

    @property
    def fitted(self):
        """
        Current fitting state.

        :type: `bool`
        """
        return self._fitted

    @fitted.setter
    def fitted(self, fitted):
        """Property-setter used to reset all fit variables."""
        _check_type(fitted, (bool, ), item_name='fitted')
        if fitted and not self._fitted:
            logger.warning(
                "The property 'fitted' can not be set to 'True' directly. "
                "Please use the .fit() method to fit the clustering "
                "algorithm.")
        elif fitted and self._fitted:
            logger.warning(
                "The property 'fitted' can not be set to 'True' directly. "
                "The clustering algorithm has already been fitted.")
        else:
            self._cluster_centers_ = None
            self._picks = None
            self._info = None
            self._fitted_data = None
            self._labels_ = None
            self._fitted = False

    # --------------------------------------------------------------------
    @staticmethod
    def _check_n_clusters(n_clusters: int):
        """Check that the number of clusters is a positive integer."""
        _check_type(n_clusters, ('int', ), item_name='n_clusters')
        if n_clusters <= 0:
            raise ValueError(
                "The number of clusters must be a positive integer. "
                f"Provided: '{n_clusters}'.")
        return n_clusters

    @staticmethod
    def _check_reject_by_annotation(reject_by_annotation):
        """Checks the reject_by_annotation argument."""
        _check_type(reject_by_annotation, (bool, str, None),
                    item_name='reject_by_annotation')
        if isinstance(reject_by_annotation, bool):
            if reject_by_annotation:
                reject_by_annotation = 'omit'
            else:
                reject_by_annotation = None
        elif isinstance(reject_by_annotation, str):
            if reject_by_annotation != 'omit':
                raise ValueError(
                    "Argument 'reject_by_annotation' only allows for 'False', "
                    "'True' (omit), or 'omit'. "
                    f"Provided: '{reject_by_annotation}'.")
        return reject_by_annotation
