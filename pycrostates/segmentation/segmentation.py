import itertools

import numpy as np
from mne import BaseEpochs
from mne.io import BaseRaw

from ..viz import plot_segmentation
from ..utils import _corr_vectors
from ..utils._checks import _check_type


def _compute_microstate_parameters(segmentation, data, maps, maps_names, sfreq,
                                   norm_gfp=True):
    """
    Compute microstate parameters.

    Parameters
    ----------
    inst : `~mne.io.Raw`, `~mne.Evoked`, list
        Instance or list of instances containing data to predict.
    modK : :class:`BaseClustering`
        Modified K-Means Clustering algorithm use to segment data
    norm_gfp : bool
        Either or not to normalize globalfield power.
    half_window_size: int
        Number of samples used for the half windows size while smoothing
        labels. Window size = 2 * half_window_size + 1
    factor: int
        Factor used for label smoothing. 0 means no smoothing.
        Defaults to 0.
    crit: float
        Converge criterion. Default to 10e-6.

    %(reject_by_annotation_raw)s
    %(verbose)s

    Returns
    ----------
    dict : list of dic
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
    'occurences' : Occurences
        Mean number of segment assigned to a given state per second. This
        metrics is expressed in segment per second ( . / s).

    """
    gfp = np.std(data, axis=0)
    if norm_gfp:
        gfp = gfp / np.linalg.norm(gfp)

    gfp = np.std(data, axis=0)
    segments = [(s, list(group))
                for s, group in itertools.groupby(segmentation)]
    d = {}
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
            occurence = len(s_segments) / len(segmentation) * sfreq
            d[f'{state_name}_occurences'] = occurence

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
            d[f'{state_name}_occurences'] = 0
    d['unlabeled'] = len(np.argwhere(segmentation == 0)) / len(gfp)
    return d


class BaseSegmentation():
    def __init__(self, segmentation, inst, cluster_centers, names=None):
        self.segmentation = segmentation
        self.inst = inst
        self.cluster_centers = cluster_centers

        if names:
            if len(self.cluster_centers) == len(names):
                self.names = names
            else:
                raise ValueError(
                    'Cluster_centers and cluster_centers_names must have the '
                    'same length')
        else:
            self.names = [f'{c+1}' for c in range(len(cluster_centers))]


class RawSegmentation(BaseSegmentation):
    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        _check_type(self.inst, (BaseRaw, ))

        data = self.inst.get_data()
        if data.shape[1] != len(self.segmentation):
            raise ValueError(
                'Instance and segmentation must have the same number of '
                'samples.')

    def plot(self, tmin: float = 0.0, tmax: float = None):
        fig, ax = plot_segmentation(segmentation=self.segmentation,
                                    inst=self.inst,
                                    cluster_centers=self.cluster_centers,
                                    names=self.names,
                                    tmin=tmin,
                                    tmax=tmax)
        return fig, ax

    def compute_parameters(self, norm_gfp=True):
        d = _compute_microstate_parameters(
            self.segmentation, self.inst.get_data(), self.cluster_centers,
            self.names, self.inst.info['sfreq'], norm_gfp=norm_gfp)
        return d


class EpochsSegmentation(BaseSegmentation):
    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        _check_type(self.inst, (BaseEpochs, ))

        data = self.inst.get_data()
        if data.shape[0] != self.segmentation.shape[0]:
            raise ValueError(
                'Epochs and segmentation must have the number of ''epochs.')
        if data.shape[2] != self.segmentation.shape[1]:
            raise ValueError(
                'Epochs and segmentation must have the number of samples per '
                'epoch.')

    def compute_parameters(self, norm_gfp=True):
        data = self.inst.get_data()
        data = np.swapaxes(data, 0, 1)
        data = data.reshape(data.shape[0], -1)
        segmentation = self.segmentation.reshape(-1)
        d = _compute_microstate_parameters(
            segmentation, data, self.cluster_centers, self.names,
            self.inst.info['sfreq'], norm_gfp=norm_gfp)
        return d
