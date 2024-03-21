"""Segmentation module for segmented data."""

from __future__ import annotations  # c.f. PEP 563, PEP 649

import itertools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.axes import Axes
from mne import BaseEpochs
from mne.io import BaseRaw
from mne.utils import check_version

from ..utils import _corr_vectors
from ..utils._checks import _check_type
from ..utils._docs import fill_doc
from ..utils._logs import logger
from ..viz import plot_cluster_centers
from .entropy import entropy
from .transitions import _compute_expected_transition_matrix, _compute_transition_matrix

if TYPE_CHECKING:
    from typing import Optional, Union

    from .._typing import ScalarFloatArray, ScalarIntArray


@fill_doc
class _BaseSegmentation(ABC):
    """Base class for a Microstates segmentation.

    Parameters
    ----------
    labels : array of shape (n_samples, ) or (n_epochs, n_samples)
        Microstates labels attributed to each sample, i.e. the segmentation.
    inst : Raw | Epochs
        MNE instance used to predict the segmentation.
    %(cluster_centers_seg)s
    %(cluster_names)s
    %(predict_parameters)s
    """

    @abstractmethod
    def __init__(
        self,
        labels: ScalarIntArray,
        inst: Union[BaseRaw, BaseEpochs],
        cluster_centers_: ScalarFloatArray,
        cluster_names: Optional[list[str]] = None,
        predict_parameters: Optional[dict] = None,
    ):
        # check input
        _check_type(labels, (np.ndarray,), "labels")
        _check_type(cluster_centers_, (np.ndarray,), "cluster_centers_")
        if cluster_centers_.ndim != 2:
            raise ValueError(
                "Argument 'cluster_centers_' should be a 2D array. The "
                f"provided array shape is {cluster_centers_.shape} which has "
                f"{cluster_centers_.ndim} dimensions."
            )

        self._labels = labels
        self._inst = inst
        self._cluster_centers_ = cluster_centers_
        self._cluster_names = _BaseSegmentation._check_cluster_names(
            cluster_names, self._cluster_centers_
        )
        self._predict_parameters = _BaseSegmentation._check_predict_parameters(
            predict_parameters
        )

    def __repr__(self) -> str:
        name = self.__class__.__name__
        s = f"<{name} | n = {len(self._cluster_centers_)} cluster centers |"
        s += f" {self._inst.__repr__()[1:-1]}>"
        return s

    def _repr_html_(self, caption=None):
        from ..html_templates import repr_templates_env  # pylint: disable=C0415

        template = repr_templates_env.get_template("BaseSegmentation.html.jinja")
        return template.render(
            name=self.__class__.__name__,
            n_clusters=len(self._cluster_centers_),
            cluster_names=self._cluster_names,
            inst_repr=self._inst._repr_html_(),
        )

    def compute_parameters(self, norm_gfp: bool = True, return_dist: bool = False):
        """Compute microstate parameters.

        .. warning::

            When working with `~mne.Epochs`, this method will put together segments of
            all epochs. This could lead to wrong interpretation especially on state
            durations. To avoid this behaviour, make sure to set the ``reject_edges``
            parameter to ``True`` when creating the segmentation.

        Parameters
        ----------
        norm_gfp : bool
            If True, the :term:`global field power` (GFP) is normalized.
        return_dist : bool
            If True, return the parameters distributions.

        Returns
        -------
        dict : dict
            Dictionaries containing microstate parameters as key/value pairs.
            Keys are named as follow: ``'{microstate name}_{parameter name}'``.

            Available parameters are listed below:

            * ``mean_corr``: Mean correlation value for each time point assigned to a
              given state.
            * ``gev``: Global explained variance expressed by a given state.
              It is the sum of global explained variance values of each time point
              assigned to a given state.
            * ``timecov``: Time coverage, the proportion of time during which
              a given state is active. This metric is expressed as a ratio.
            * ``meandurs``: Mean durations of segments assigned to a given
              state. This metric is expressed in seconds (s).
            * ``occurrences``: Occurrences per second, the mean number of
              segment assigned to a given state per second. This metrics is expressed
              in segment per second.
            * ``dist_corr`` (req. ``return_dist=True``): Distribution of
              correlations values of each time point assigned to a given state.
            * ``dist_gev`` (req. ``return_dist=True``): Distribution of global
              explained variances values of each time point assigned to a given state.
            * ``dist_durs`` (req. ``return_dist=True``): Distribution of
              durations of each segments assigned to a given state. Each value is
              expressed in seconds (s).
        """
        _check_type(norm_gfp, (bool,), "norm_gfp")
        _check_type(return_dist, (bool,), "return_dist")

        # retrieve sampling frequency for convenience
        sfreq = self._inst.info["sfreq"]

        # don't copy the data/labels array, get_data, swapaxes, reshape are
        # returning a new view of the array, which is fine since we do not modify it.
        labels = self._labels  # same pointer, no memory overhead.
        if isinstance(self._inst, BaseRaw):
            data = self._inst.get_data()
            # sanity-checks
            assert labels.ndim == 1
            assert data.ndim == 2
            assert labels.size == data.shape[1]
        elif isinstance(self._inst, BaseEpochs):
            kwargs_epochs = dict(copy=False) if check_version("mne", "1.6") else dict()
            data = self._inst.get_data(**kwargs_epochs)
            # sanity-checks
            assert labels.ndim == 2
            assert data.ndim == 3
            assert labels.size == data.shape[0] * data.shape[2]
            # create a 2D view of the data array
            data = np.swapaxes(data, 0, 1)
            data = data.reshape(data.shape[0], -1)
            # create a 1D view of the labels array
            labels = labels.reshape(-1)

        gfp = np.std(data, axis=0)
        if norm_gfp:
            labeled = np.argwhere(labels != -1)  # ignore unlabeled segments
            gfp /= np.linalg.norm(gfp[labeled])  # normalize

        segments = [(s, list(group)) for s, group in itertools.groupby(labels)]

        params = dict()
        for s, state in enumerate(self._cluster_centers_):
            state_name = self._cluster_names[s]
            arg_where = np.argwhere(labels == s)
            if len(arg_where) != 0:
                labeled_tp = data.T[arg_where][:, 0, :].T
                labeled_gfp = gfp[arg_where][:, 0]
                state_array = np.array([state] * len(arg_where)).transpose()

                dist_corr = _corr_vectors(state_array, labeled_tp)
                params[f"{state_name}_mean_corr"] = np.mean(np.abs(dist_corr))
                dist_gev = (labeled_gfp * dist_corr) ** 2 / np.sum(gfp**2)
                params[f"{state_name}_gev"] = np.sum(dist_gev)

                s_segments = np.array([len(group) for s_, group in segments if s_ == s])
                occurrences = len(s_segments) / len(np.where(labels != -1)[0]) * sfreq
                params[f"{state_name}_occurrences"] = occurrences

                timecov = np.sum(s_segments) / len(np.where(labels != -1)[0])
                params[f"{state_name}_timecov"] = timecov

                dist_durs = s_segments / sfreq
                params[f"{state_name}_meandurs"] = np.mean(dist_durs)

                if return_dist:
                    params[f"{state_name}_dist_corr"] = dist_corr
                    params[f"{state_name}_dist_gev"] = dist_gev
                    params[f"{state_name}_dist_durs"] = dist_durs

            else:
                params[f"{state_name}_mean_corr"] = 0.0
                params[f"{state_name}_gev"] = 0.0
                params[f"{state_name}_timecov"] = 0.0
                params[f"{state_name}_meandurs"] = 0.0
                params[f"{state_name}_occurrences"] = 0.0

                if return_dist:
                    params[f"{state_name}_dist_corr"] = np.array([], dtype=float)
                    params[f"{state_name}_dist_gev"] = np.array([], dtype=float)
                    params[f"{state_name}_dist_durs"] = np.array([], dtype=float)

        params["unlabeled"] = len(np.argwhere(labels == -1)) / len(gfp)
        return params

    def compute_transition_matrix(self, stat="probability", ignore_repetitions=True):
        """Compute the observed transition matrix.

        Count the number of transitions from one state to another and aggregate the
        result as statistic. Transition "from" and "to" unlabeled segments ``-1`` are
        ignored.

        Parameters
        ----------
        %(stat_transition)s
        %(ignore_repetitions)s

        Returns
        -------
        %(transition_matrix)s

        Notes
        -----
        .. warning::

            When working with `~mne.Epochs`, this method will take into account
            transitions that occur between epochs. This could lead to wrong
            interpretation when working with discontinuous data. To avoid this
            behaviour, make sure to set the ``reject_edges`` parameter to ``True`` when
            predicting the segmentation.
        """
        return _compute_transition_matrix(
            self._labels,
            self._cluster_centers_.shape[0],
            stat,
            ignore_repetitions,
        )

    @fill_doc
    def compute_expected_transition_matrix(
        self, stat="probability", ignore_repetitions=True
    ):
        """Compute the expected transition matrix.

        Compute the theoretical transition matrix as if time course was ignored, but
        microstate proportions kept (i.e. shuffled segmentation). This matrix can be
        used to quantify/correct the effect of microstate time coverage on the observed
        transition matrix obtained with the method
        ``compute_expected_transition_matrix``.
        Transition "from" and "to" unlabeled segments ``-1`` are ignored.

        Parameters
        ----------
        %(stat_expected_transitions)s
        %(ignore_repetitions)s

        Returns
        -------
        %(transition_matrix)s
        """
        return _compute_expected_transition_matrix(
            self._labels,
            n_clusters=self._cluster_centers_.shape[0],
            stat=stat,
            ignore_repetitions=ignore_repetitions,
        )

    # Information ------------------------------------------------------------
    @fill_doc
    def entropy(
        self,
        ignore_repetitions: bool = False,
        log_base: Union[float, str] = 2,
    ):
        r"""Compute the Shannon entropy of the segmentation.

        Compute the Shannon entropy\ :footcite:p:`shannon1948mathematical`
        of the microstate symbolic sequence.

        Parameters
        ----------
        %(ignore_repetitions)s
        %(log_base)s

        Returns
        -------
        h : float
            The Shannon entropy.

        References
        ----------
        .. footbibliography::
        """
        return entropy(
            self,
            ignore_repetitions=ignore_repetitions,
            log_base=log_base,
        )

    @fill_doc
    def plot_cluster_centers(
        self,
        axes: Optional[Union[Axes,]] = None,
        *,
        block: bool = False,
        show: Optional[bool] = None,
    ):
        """Plot cluster centers as topographic maps.

        Parameters
        ----------
        %(axes_topo)s
        %(block)s
        %(show)s

        Returns
        -------
        fig : Figure
            Matplotlib figure containing the topographic plots.
        """
        return plot_cluster_centers(
            self._cluster_centers_,
            self._inst.info,
            self._cluster_names,
            axes,
            block=block,
            show=show,
        )

    # --------------------------------------------------------------------
    @staticmethod
    def _check_cluster_names(
        cluster_names: list[str],
        cluster_centers_: ScalarFloatArray,
    ):
        """Check that the argument 'cluster_names' is valid."""
        _check_type(cluster_names, (list, None), "cluster_names")
        if cluster_names is None:
            return [str(k) for k in range(1, len(cluster_centers_) + 1)]
        else:
            if len(cluster_centers_) == len(cluster_names):
                return cluster_names
            else:
                raise ValueError(
                    "The same number of cluster centers and cluster names "
                    f"should be provided. There are {len(cluster_centers_)} "
                    f"cluster centers and '{len(cluster_names)}' provided."
                )

    @staticmethod
    def _check_predict_parameters(predict_parameters: dict):
        """Check that the argument 'predict_parameters' is valid."""
        _check_type(predict_parameters, (dict, None), "predict_parameters")
        if predict_parameters is None:
            return None
        # valid keys from pycrostates prediction
        valid_keys = (
            "factor",
            "tol",
            "half_window_size",
            "min_segment_length",
            "reject_edges",
            "reject_by_annotation",
        )
        # Let the door open for custom prediction with different keys, so log
        # a warning instead of raising.
        for key in predict_parameters.keys():
            if key not in valid_keys:
                logger.warning(
                    "The key '%s' in predict_parameters is not part of "
                    "the default set of keys supported by pycrostates.",
                    key,
                )
        return predict_parameters

    # --------------------------------------------------------------------
    @property
    def predict_parameters(self) -> dict:
        """Parameters used to predict the current segmentation.

        :type: `dict`
        """
        if self._predict_parameters is None:
            logger.info(
                "Argument 'predict_parameters' was not provided when creating "
                "the segmentation."
            )
            return None
        return self._predict_parameters.copy()

    @property
    def labels(self) -> ScalarIntArray:
        """Microstate label attributed to each sample (the segmentation).

        :type: `~numpy.array`
        """
        return self._labels.copy()

    @property
    def cluster_centers_(self) -> ScalarFloatArray:
        """Cluster centers (i.e topographies) used to compute the segmentation.

        :type: `~numpy.array`
        """
        return self._cluster_centers_.copy()

    @property
    def cluster_names(self) -> list[str]:
        """Name of the cluster centers.

        :type: `list`
        """
        return self._cluster_names.copy()
