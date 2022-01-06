from abc import ABC, abstractmethod
from copy import copy, deepcopy

from mne import BaseEpochs
from mne.io import BaseRaw, RawArray

from ..utils._logs import verbose
from ..utils._checks import _check_type, _check_value, _check_n_jobs


class _BaseCluster(ABC):
    """
    Base Class for Microstates Clustering algorithms.
    """

    @abstractmethod
    def __init__(self):
        self._fitted = False
        self._n_clusters = None
        self._clusters_names = None
        self._cluster_centers = None

    def __repr__(self) -> str:
        name = self.__class__.__name__
        if self.n_clusters is not None:
            s = f'<{name} | fitted on n = {self.n_clusters} cluster centers>'
        else:
            s = f'<{name} | not fitted>
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
            raise ValueError(
                'Clustering algorithm must be fitted before using '
                f'{self.__class__.__name__}')

    @abstractmethod
    @fill_doc
    @verbose
    def fit(self, inst, tmin=None, tmax=None, reject_by_annotation=True,
            n_jobs=1, *, verbose=None):
        """Segment `~mne.io.Raw` or `~mne.Epochs` instance into microstate
        sequence.

        Parameters
        ----------
        %(fit_inst)s
        %(raw_tmin)s
        %(raw_tmax)s
        %(reject_by_annotation_raw)s
        %(n_jobs)s
        %(verbose)s
        """
        _check_type(inst, (BaseRaw, BaseEpochs), item_name='inst')
        _check_type(tmin, (None, 'numeric'), item_name='tmin')
        _check_type(tmax, (None, 'numeric'), item_name='tmax')
        _check_type(reject_by_annotation, (bool, ),
                    item_name='reject_by_annotation')
        n_jobs = _check_n_jobs(n_jobs)

        # TODO: Add check for tmin/tmax values

    def rename_clusters(self, mapping: dict):
        """
        Rename the clusters in-place.

        Parameters
        ----------
        mapping : dict
            Mapping from the old names to the new names.
            key: old name, value: new name.
        """
        # TODO: Add support for mapping to be a callable function (str -> str)
        self._check_fit()
        _check_type(mapping, (dict, ), item_name='mapping')
        for key in mapping:
            _check_value(key, self.clusters_names, item_name='old name')
        for value in mapping.values:
            _check_type(value, ('str', ), item_name='new name')
        self.clusters_names = [mapping[name] if name in mapping else name
                               for name in self._clusters_names]

    def reorder_clusters(self, mapping: dict):
        """
        Reorder the clusters in-place.

        Parameters
        ----------
        mapping : dict
            Mapping from the old order to the new order. The positions are
            0-indexed. key: old position, value: new position.
        """
        self._check_fit()
        _check_type(mapping, (dict, ), item_name='mapping')
        valids = tuple(range(len(self.cluster_centers)))
        for key in mapping:
            _check_value(key, valids, item_name='old position')
        for value in mapping.values():
            _check_value(value, valids, item_name='new position')

        self.cluster_centers_ = self.cluster_centers_[order]
        self.names = [self.names[o] for o in order]

    @abstractmethod
    @fill_doc
    @verbose
    def predict(self, inst, factor=0, half_window_size=3, tol=10e-6,
                min_segment_lenght=0, reject_edges=True,
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
            Converge tolerance.
        min_segment_lenght : int
            Minimum segment length (in samples). If a segment is shorter than
            this value, it will be recursively reasigned to neighbouring
            segments based on absolute spatial correlation.
        reject_edges : bool
            If True, set first and last segments to unlabeled.
        %(reject_by_annotation_raw)s
        %(verbose)s
        """
        self._check_fit()
        _check_type(inst, (BaseRaw, BaseEpochs), item_name='inst')
        _check_type(factor, ('int', ), item_name='factor')
        _check_type(half_window_size, ('int', ), item_name='half_window_size')
        _check_type(tol, ('numeric', ), item_name='tol')
        _check_type(min_segment_lenght, ('int', ),
                    item_name='min_segment_lenght')
        _check_type(reject_edges, (bool, ), item_name='reject_edges')
        _check_type(reject_by_annotation, (bool, ),
                    item_name='reject_by_annotation')

    # --------------------------------------------------------------------
    @property
    def fitted(self):
        """
        Current fitting state.

        :type: `bool`
        """
        return self._fitted

    @property
    def n_clusters(self):
        """
        Number of clusters. Returns None if cluster algorithm has not been
        fitted.

        :type: `int`
        """
        return self._n_clusters

    @property
    def clusters_names(self):
        """
        Name of the clusters. Returns None if cluster algorithm has not been
        fitted.

        :type: `list`
        """
        return self._clusters_names

    @property
    def cluster_centers(self):
        """
        Center of the clusters. Returns None if cluster algorithm has not been
        fitted.

        :type: `~numpy.array`
        """
        return self._cluster_centers
