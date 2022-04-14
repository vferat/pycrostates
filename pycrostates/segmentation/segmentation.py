from abc import ABC, abstractmethod
import itertools

import numpy as np
from mne import BaseEpochs, pick_info
from mne.io import BaseRaw

from ..utils import _corr_vectors
from ..utils._checks import _check_type
from ..utils._docs import fill_doc
from ..utils._logs import logger
from ..viz import (
    plot_raw_segmentation, plot_epoch_segmentation, plot_cluster_centers)


def _compute_microstate_parameters(
        labels,
        data,
        maps,
        maps_names,
        sfreq,
        norm_gfp: bool = True):
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
                for s, group in itertools.groupby(labels)]
    d = dict()
    for s, state in enumerate(maps):
        state_name = maps_names[s]
        arg_where = np.argwhere(labels == s)
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
                [len(group) for s_, group in segments if s_ == s])
            occurrences = (len(s_segments)
                           / len(np.where(labels != -1)[0])
                           * sfreq)

            d[f'{state_name}_occurrences'] = occurrences

            timecov = np.sum(s_segments) / len(np.where(labels != -1)[0])
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
    d['unlabeled'] = len(np.argwhere(labels == -1)) / len(gfp)
    return d


class _BaseSegmentation(ABC):
    """
    Base class for a Microstates segmentation.
    """

    @abstractmethod
    def __init__(
            self,
            labels,
            inst,
            cluster_centers_,
            cluster_names=None,
            predict_parameters=None,
            ):
        # check input
        _check_type(labels, (np.ndarray, ), 'labels')
        _check_type(cluster_centers_, (np.ndarray, ), 'cluster_centers_')
        if cluster_centers_.ndim != 2:
            raise ValueError(
                "Argument 'cluster_centers_' should be a 1D array.")

        self._labels = labels
        self._inst = inst
        self._cluster_centers_ = cluster_centers_
        self._cluster_names = _BaseSegmentation._check_cluster_names(
            cluster_names, self._cluster_centers_)
        self._predict_parameters = _BaseSegmentation._check_predict_parameters(
            predict_parameters)
        # sanity-check
        assert self._inst.times.size == self._labels.shape[-1]

    def __repr__(self) -> str:
        name = self.__class__.__name__
        s = f'<{name} | n = {len(self._cluster_centers_)} cluster centers |'
        s += f' {self._inst.__repr__()[1:-1]}>'
        return s

    def _repr_html_(self, caption=None):
        from ..html_templates import repr_templates_env
        template = repr_templates_env.get_template(
            'BaseSegmentation.html.jinja')
        return template.render(
            name=self.__class__.__name__,
            n_clusters=len(self._cluster_centers_),
            cluster_names=self._cluster_names,
            inst_repr=self._inst._repr_html_(),
            )

    def plot_cluster_centers(self, axes=None, block=False):
        """
        Plot cluster centers as topographic maps.

        Parameters
        ----------
        axes : None | Axes
            Either none to create a new figure or axes (or an array of axes)
            on which the topographic map should be plotted.
        block : bool
            Whether to halt program execution until the figure is closed.

        Returns
        -------
        f : Figure
            Matplotlib figure containing the topographic plots.
        """
        return plot_cluster_centers(self._cluster_centers_, self._inst.info,
                                    self._cluster_names, axes, block)

    # --------------------------------------------------------------------
    @staticmethod
    def _check_cluster_names(cluster_names, cluster_centers_):
        """
        Checks that the argument 'cluster_names' is valid.
        """
        _check_type(cluster_names, (list, None), 'cluster_names')
        if cluster_names is None:
            return [str(k) for k in range(1, len(cluster_centers_) + 1)]
        else:
            if len(cluster_centers_) == len(cluster_names):
                return cluster_names
            else:
                raise ValueError(
                    "The same number of cluster centers and cluster names "
                    f"should be provided. There are {len(cluster_centers_)} "
                    f"cluster centers and '{len(cluster_names)}' provided.")

    @staticmethod
    def _check_predict_parameters(predict_parameters):
        """
        Checks that the argument 'predict_parameters' is valid.
        """
        _check_type(predict_parameters, (dict, None), 'predict_parameters')
        if predict_parameters is None:
            return None
        # valid keys from pycrostates prediction
        valid_keys = (
            'factor',
            'tol',
            'half_window_size',
            'min_segment_length',
            'reject_edges',
            'reject_by_annotation',
            )
        # Let the door open for custom prediction with different keys, so log
        # a warning instead of raising.
        for key in predict_parameters.keys():
            if key not in valid_keys:
                logger.warning(
                    f"The key '{key}' in predict_parameters is not part of "
                    "the default set of keys supported by pycrostates.")
        return predict_parameters

    # --------------------------------------------------------------------
    @property
    def predict_parameters(self):
        """
        Parameters used to predict the current segmentation.

        :type: `dict`
        """
        if self._predict_parameters:
            return self._predict_parameters.copy()
        else:
            logger.info(
                'predict_parameters was not provided when creating the '
                'segmentation. Returning None.')
            return None

    @property
    def labels(self):
        """
        Segmentation predicted.
        """
        return self._labels.copy()

    @property
    def cluster_centers_(self):
        """
        Center of the clusters.

        :type: `~numpy.array`
        """
        return self._cluster_centers_.copy()

    @property
    def cluster_names(self):
        """
        Name of the clusters.

        :type: `list`
        """
        return self._cluster_names.copy()

# TODO: Parameters to be added to docdict
class RawSegmentation(_BaseSegmentation):
    """
    Contains the segmentation for a raw instance.

    Parameters
    ----------
    labels : array
    inst : Raw
    cluster_centers_ : array
    cluster_names : list
    predict_parameters : dict
    """

    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        _check_type(self._inst, (BaseRaw, ), item_name='raw')
        if self._labels.ndim != 1:
            raise ValueError("Argument 'labels' should be a 1D array.")

    @fill_doc
    def plot(
            self,
            tmin=0.0,  # TODO: Should be None by default
            tmax=None,
            cmap=None,
            axes=None,
            cbar_axes=None,
            block: bool = False,
            verbose=None
            ):
        """
        Plot the segmentation.

        Parameters
        ----------
        %(tmin_raw)s
        %(tmax_raw)s
        cmap : matplotlib colormap name
            The mapping from label name to color space.
        axes : None | Axes
            Either none to create a new figure or axes on which the
            segmentation is plotted.
        cbar_axes : None | Axes
            Axes on which to draw the colorbar, otherwise the colormap takes
            space from the main axes.
        block : bool
            Whether to halt program execution until the figure is closed.
        %(verbose)s
        
        Returns
        -------
        fig : Figure
            Matplotlib figure containing the segmentation.
        """
        # Error checking on the input is performed in the viz function.
        return plot_raw_segmentation(
            labels=self._labels,
            raw=self.raw,
            n_clusters=self._cluster_centers_.shape[0],
            cluster_names=self._cluster_names,
            tmin=tmin,
            tmax=tmax,
            cmap=cmap,
            axes=axes,
            cbar_axes=cbar_axes,
            block=block,
            verbose=verbose
            )

    def compute_parameters(
            self,
            norm_gfp: bool = True
            ):
        """
        Compute microstate parameters.

        Parameters
        ----------
        norm_gfp : bool
            Either or not to normalized global field power.

        Returns
        -------
        dict : dict
            Dictionaries containing microstate parameters as key/value pairs.
            Keys are named as follow: '{microstate name}_{parameter}'.

            Available parameters are list below:
            'dist_corr': Distribution of correlations
                Correlation values of each time point assigned to a given
                state.
            'mean_corr': Mean correlation
                Mean correlation value of each time point assigned to a given
                state.
            'dist_gev': Distribution of global explained variances
                Global explained variance values of each time point assigned to
                a given state.
            'gev':  Global explained variance
                Total explained variance expressed by a given state.
                It is the sum of global explained variance values of each
                time point assigned to a given state.
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
            self._labels,
            self._inst.get_data(),
            self._cluster_centers_,
            self._cluster_names,
            self.raw.info['sfreq'],
            norm_gfp=norm_gfp,
            )

    # --------------------------------------------------------------------
    @property
    def raw(self):
        """
        Raw instance.
        """
        return self._inst


# TODO: Parameters to be added to docdict
class EpochsSegmentation(_BaseSegmentation):
    """
    Contains the segmentation for an epoch instance.

    Parameters
    ----------
    labels : array
    inst : Epochs
    cluster_centers_ : array
    cluster_names : list
    predict_parameters : dict
    """

    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        _check_type(self._inst, (BaseEpochs, ), 'epochs')
        if self._labels.ndim != 2:
            raise ValueError("Argument 'labels' should be a 2D array.")

        # sanity-check
        assert len(self._inst) == self._labels.shape[0]

    def compute_parameters(
            self,
            norm_gfp: bool = True
            ):
        """
        Compute microstate parameters.

        Parameters
        ----------
        norm_gfp : bool
            Either or not to normalized global field power.

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
        data = self._inst.get_data()
        data = np.swapaxes(data, 0, 1)
        data = data.reshape(data.shape[0], -1)
        labels = self._labels.copy().reshape(-1)
        return _compute_microstate_parameters(
            labels,
            data,
            self._cluster_centers_,
            self._cluster_names,
            self.epochs.info['sfreq'],
            norm_gfp=norm_gfp,
            )

    @fill_doc
    def plot(
            self,
            cmap=None,
            axes=None,
            cbar_axes=None,
            block: bool = False,
            verbose=None
            ):
        """
        Plot segmentation.

        Parameters
        ----------
        cmap : matplotlib colormap name
            The mapping from label name to color space.
        axes : None | Axes
            Either none to create a new figure or axes on which the
            segmentation is plotted.
        cbar_axes : None | Axes
            Axes on which to draw the colorbar, otherwise the colormap takes
            space from the main axes.
        block : bool
            Whether to halt program execution until the figure is closed.
        %(verbose)s

        Returns
        -------
        fig : Figure
            Matplotlib figure containing the segmentation.
        """
        # Error checking on the input is performed in the viz function.
        return plot_epoch_segmentation(
            labels=self._labels,
            epochs=self.epochs,
            n_clusters=self._cluster_centers_.shape[0],
            cluster_names=self._cluster_names,
            cmap=cmap,
            axes=axes,
            cbar_axes=cbar_axes,
            block=block,
            verbose=verbose
            )

    # --------------------------------------------------------------------
    @property
    def epochs(self):
        """
        Epochs instance.
        """
        return self._inst
