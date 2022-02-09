from abc import ABC, abstractmethod
import itertools

from mne import BaseEpochs
from mne.io import BaseRaw
import numpy as np

from ..utils import _corr_vectors
from ..utils._checks import _check_type
from ..utils._docs import fill_doc
from ..viz import plot_segmentation


def _compute_microstate_parameters(segmentation, data, maps, maps_names, sfreq,
                                   norm_gfp=True):
    """
    Compute microstate parameters.

    Returns
    -------
    dict : list of dict
        Dictionaries containing microstate metrics.

    Attributes
    ----------
    'dist_corr': Distribution of correlations
        Correlation values of each time point assigned to a given state.
    'mean_corr': Mean correlation
        Mean correlation value of each time point assigned to a given state.
    'dist_gev': Distribution of global explained variances
        Global explained variance values of each time point assigned to a given
        state.
    'gev':  Global explained variance
        Total explained variance expressed by a given state. It is the sum of
        global explained variance values of each time point assigned to a given
        state.
    'timecov': Time coverage
        The proportion of time during which a given state is active. This
        metric is expressed in percentage (%%).
    'dist_durs': Distribution of durations.
        Duration of each segments assigned to a given state. Each value is
        expressed in seconds (s).
    'meandurs': Mean duration
        Mean temporal duration segments assigned to a given state. This metric
        is expressed in seconds (s).
    'occurrences' : occurrences
        Mean number of segment assigned to a given state per second. This
        metrics is expressed in segment per second ( . / s).

    """
    gfp = np.std(data, axis=0)
    if norm_gfp:
        gfp = gfp / np.linalg.norm(gfp)

    gfp = np.std(data, axis=0)
    segments = [(s, list(group))
                for s, group in itertools.groupby(segmentation)]
    d = dict()
    for s, state in enumerate(maps):
        state_name = maps_names[s]
        arg_where = np.argwhere(segmentation == s+1)
        if len(arg_where) != 0:
            labeled_tp = data.T[arg_where][:, 0, :].T
            labeled_gfp = gfp[arg_where][:, 0]
            state_array = np.array([state]*len(arg_where)).transpose()

            corr = _corr_vectors(state_array, labeled_tp)
            d[f'{state_name}_dist_corr'] = corr
            d[f'{state_name}_mean_corr'] = np.mean(np.abs(corr))

            gev = (labeled_gfp * corr) ** 2 / np.sum(gfp ** 2)
            d[f'{state_name}_dist_gev'] = gev
            d[f'{state_name}_gev'] = np.sum(gev)

            s_segments = np.array(
                [len(group) for s_, group in segments if s_ == s + 1])
            occurrences = len(s_segments) / len(segmentation) * sfreq
            d[f'{state_name}_occurrences'] = occurrences

            timecov = np.sum(s_segments) / len(np.where(segmentation != 0)[0])
            d[f'{state_name}_timecov'] = timecov

            durs = s_segments / sfreq
            d[f'{state_name}_dist_durs'] = durs
            d[f'{state_name}_meandurs'] = np.mean(durs)
        else:
            d[f'{state_name}_dist_corr'] = 0
            d[f'{state_name}_mean_corr'] = 0
            d[f'{state_name}_dist_gev'] = 0
            d[f'{state_name}_gev'] = 0
            d[f'{state_name}_timecov'] = 0
            d[f'{state_name}_dist_durs'] = 0
            d[f'{state_name}_meandurs'] = 0
            d[f'{state_name}_occurrences'] = 0
    d['unlabeled'] = len(np.argwhere(segmentation == 0)) / len(gfp)
    return d


class _BaseSegmentation(ABC):
    """
    Base class for a Microstates segmentation.
    """

    @abstractmethod
    def __init__(self, segmentation, inst, picks, cluster_centers_,
                 clusters_names=None):
        self._segmentation = _BaseSegmentation._check_segmentation(
            segmentation)
        self._inst = inst
        self._picks = picks
        self._cluster_centers_ = cluster_centers_
        self._clusters_names = _BaseSegmentation._check_cluster_names(
            clusters_names, cluster_centers_)

        # sanity-check
        assert self._inst.times.size == self._segmentation.shape[-1]

    # --------------------------------------------------------------------
    @staticmethod
    def _check_segmentation(segmentation):
        """
        Checks that the argument 'segmentation' is valid.
        """
        return np.array(segmentation)

    @staticmethod
    def _check_cluster_names(clusters_names, cluster_centers_):
        """
        Checks that the argument 'cluster_names' is valid.
        """
        if clusters_names is None:
            return [str(k) for k in range(1, len(cluster_centers_)+1)]
        else:
            if len(cluster_centers_) == len(clusters_names):
                return clusters_names
            else:
                raise ValueError(
                    "The same number of cluster centers and cluster names "
                    f"should be provided. There are {len(cluster_centers_)} "
                    "cluster centers and '{len(cluster_names)}' provided.")

    # --------------------------------------------------------------------
    @property
    def segmentation(self):
        """
        Segmentation predicted.
        """
        return self._segmentation

    @property
    def picks(self):
        """
        Picks used to fit the clustering algorithm and to predict the
        segmentation.

        :type: `~numpy.array`
        """
        return self._picks

    @property
    def cluster_centers_(self):
        """
        Center of the clusters.

        :type: `~numpy.array`
        """
        return self._cluster_centers_

    @property
    def clusters_names(self):
        """
        Name of the clusters.

        :type: `list`
        """
        return self._clusters_names


class RawSegmentation(_BaseSegmentation):
    """
    Contains the segmentation for a raw instance.
    """

    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        _check_type(self._inst, (BaseRaw, ), item_name='raw')

    @fill_doc
    def plot(self, tmin=0.0, tmax=None):
        """
        Plot segmentation.

        Parameters
        ----------
        %(raw_tmin)s
        %(raw_tmax)s

        Returns
        -------
        fig : :class:`matplotlib.figure.Figure`
            Figure
        ax : :class:`matplotlib.axes.Axes`
            Axis
        """
        return plot_segmentation(
            segmentation=self.segmentation, inst=self.raw,
            cluster_centers=self.cluster_centers_, names=self.clusters_names,
            tmin=tmin, tmax=tmax)

    def compute_parameters(self, norm_gfp=True):
        """
        Compute microstate parameters.

        Parameters
        ----------
        norm_gfp: bool
            Either or not to normalized global field power.
            Defaults to True.

        Returns
        -------
        dict : dict
            Dictionaries containing microstate parameters as key/value pairs.
            Keys are named as follow: '{microstate name}_{parameter}'.

            Available parameters are list below:
            'dist_corr': Distribution of correlations
                Correlation values of each time point
                assigned to a given state.
            'mean_corr': Mean correlation
                Mean correlation value of each time point
                assigned to a given state.
            'dist_gev': Distribution of global explained variances
                Global explained variance values of each time point
                assigned to a given state.
            'gev':  Global explained variance
                Total explained variance expressed by a given state.
                It is the sum of global explained variance values of each
                time point assigned to a given
                state.
            'timecov': Time coverage
                The proportion of time during which a given state is active.
                This metric is expressed in percentage (%%).
            'dist_durs': Distribution of durations.
                Duration of each segments assigned to a given state.
                Each value is expressed in seconds (s).
            'meandurs': Mean duration
                Mean temporal duration segments assigned to a given state.
                This metric is expressed in seconds (s).
            'occurrences' : occurrences
                Mean number of segment assigned to a given state per second.
                This metrics is expressed in segment per second ( . / s).
        """
        return _compute_microstate_parameters(
            self.segmentation, self.raw.get_data(picks=self.picks),
            self.cluster_centers_, self.clusters_names,
            self.raw.info['sfreq'], norm_gfp=norm_gfp)

    # --------------------------------------------------------------------
    @property
    def raw(self):
        """
        Raw instance.
        """
        return self._inst


class EpochsSegmentation(_BaseSegmentation):
    """
    Contains the segmentation for an epoch instance.
    """

    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        _check_type(self._inst, (BaseEpochs, ), 'epochs')

        # sanity-check
        assert len(self._inst) == self.segmentation.shape[0]

    def compute_parameters(self, norm_gfp=True):
        """
        Compute microstate parameters.

        Parameters
        ----------
        norm_gfp: bool
            Either or not to normalized global field power.
            Defaults to True.

        Returns
        -------
        dict : dict
            Dictionaries containing microstate parameters
            as key/value pairs.
            Keys are following named as follow:
            '{microstate name}_{parameter}'.
            Available parameters are list below:

            'dist_corr': Distribution of correlations
                Correlation values of each time point
                assigned to a given state.
            'mean_corr': Mean correlation
                Mean correlation value of each time point
                assigned to a given state.
            'dist_gev': Distribution of global explained variances
                Global explained variance values of each time point
                assigned to a given state.
            'gev':  Global explained variance
                Total explained variance expressed by a given state.
                It is the sum of global explained variance values of each
                time point assigned to a given
                state.
            'timecov': Time coverage
                The proportion of time during which a given state is active.
                This metric is expressed in percentage (%%).
            'dist_durs': Distribution of durations.
                Duration of each segments assigned to a given state.
                Each value is expressed in seconds (s).
            'meandurs': Mean duration
                Mean temporal duration segments assigned to a given state.
                This metric is expressed in seconds (s).
            'occurrences' : occurrences
                Mean number of segment assigned to a given state per second.
                This metrics is expressed in segment per second ( . / s).
        """
        data = self.epochs.get_data(picks=self.picks)
        data = np.swapaxes(data, 0, 1)
        data = data.reshape(data.shape[0], -1)
        segmentation = self.segmentation.reshape(-1)
        return _compute_microstate_parameters(
            segmentation, data, self.cluster_centers_, self.clusters_names,
            self.epochs.info['sfreq'], norm_gfp=norm_gfp)

    # --------------------------------------------------------------------
    @property
    def epochs(self):
        """
        Epochs instance.
        """
        return self._inst
