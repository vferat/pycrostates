"""Segmentation module for segmented data."""

import itertools
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
from matplotlib.axes import Axes
from mne import BaseEpochs
from mne.io import BaseRaw
from numpy.typing import NDArray

from ..utils import _corr_vectors
from ..utils._checks import _check_type
from ..utils._docs import fill_doc
from ..utils._logs import logger
from ..viz import (
    plot_cluster_centers,
    plot_epoch_segmentation,
    plot_raw_segmentation,
)


@fill_doc
class _BaseSegmentation(ABC):
    """Base class for a Microstates segmentation.

    Parameters
    ----------
    labels : Array (n_samples, ) or (n_epochs, n_samples)
        Microstates labels attributed to each sample, i.e. the segmentation.
    inst : Raw | Epochs
        MNE instance used to predict the segmentation.
    %(cluster_centers)s
    %(cluster_names)s
    %(predict_parameters)s
    """

    @abstractmethod
    def __init__(
        self,
        labels: NDArray[int],
        inst: Union[BaseRaw, BaseEpochs],
        cluster_centers_: NDArray[float],
        cluster_names: Optional[List[str]] = None,
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
        from ..html_templates import (  # pylint: disable=C0415
            repr_templates_env,
        )

        template = repr_templates_env.get_template(
            "BaseSegmentation.html.jinja"
        )
        return template.render(
            name=self.__class__.__name__,
            n_clusters=len(self._cluster_centers_),
            cluster_names=self._cluster_names,
            inst_repr=self._inst._repr_html_(),
        )

    def compute_parameters(
        self, norm_gfp: bool = True, return_dist: bool = False
    ):
        """Compute microstate parameters.

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
            Keys are named as follow: ``'{microstate name}_{parameter}'``.

            Available parameters are listed below:
                'mean_corr' : Mean correlation
                    Mean correlation value of each time point assigned to a
                    given state.
                'gev' : Global explained variance
                    Total explained variance expressed by a given state.
                    It is the sum of global explained variance values of each
                    time point assigned to a given state.
                'timecov' : Time coverage
                    The proportion of time during which a given state is
                    active. This metric is expressed in percentage (%%).
                'meandurs' : Mean duration
                    Mean temporal duration segments assigned to a given state.
                    This metric is expressed in seconds (s).
                'occurrences' : occurrences
                    Mean number of segment assigned to a given state per
                    second. This metrics is expressed in segment per second
                    ( . / s).
            If return_dist is set to True, also return the following
            distributions:
                'dist_corr' : Distribution of correlations
                    Correlation values of each time point assigned to a given
                    state.
                'dist_gev' : Distribution of global explained variances
                    Global explained variance values of each time point
                    assigned to a given state.
                'dist_durs' : Distribution of durations.
                    Duration of each segments assigned to a given state.
                    Each value is expressed in seconds (s).
        """
        return _BaseSegmentation._compute_microstate_parameters(
            self._labels.copy(),
            self._inst.get_data(),
            self._cluster_centers_,
            self._cluster_names,
            self._inst.info["sfreq"],
            norm_gfp=norm_gfp,
            return_dist=return_dist,
        )

    @fill_doc
    def plot_cluster_centers(
        self, axes: Optional[Axes] = None, *, block: bool = False
    ):
        """Plot cluster centers as topographic maps.

        Parameters
        ----------
        %(axes_topo)s
        %(block)s

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
        )

    # --------------------------------------------------------------------
    @staticmethod
    def _check_cluster_names(
        cluster_names: List[str],
        cluster_centers_: NDArray[float],
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
                    f"The key '{key}' in predict_parameters is not part of "
                    "the default set of keys supported by pycrostates."
                )
        return predict_parameters

    @staticmethod
    def _compute_microstate_parameters(
        labels: NDArray[int],
        data: NDArray[float],
        maps: NDArray[float],
        maps_names: List[str],
        sfreq: Union[int, float],
        norm_gfp: bool = True,
        return_dist: bool = False,
    ):
        """Compute microstate parameters."""
        assert (data.ndim == 3 and labels.ndim == 2) or (
            data.ndim == 2 and labels.ndim == 1
        )
        if data.ndim == 3:  # epochs
            data = np.swapaxes(data, 0, 1)
            data = data.reshape(data.shape[0], -1)
            labels = labels.reshape(-1)

        _check_type(norm_gfp, (bool,), "norm_gfp")
        _check_type(return_dist, (bool,), "return_dist")

        gfp = np.std(data, axis=0)
        if norm_gfp:
            gfp /= np.linalg.norm(gfp)

        segments = [(s, list(group)) for s, group in itertools.groupby(labels)]

        d = {}
        for s, state in enumerate(maps):
            state_name = maps_names[s]
            arg_where = np.argwhere(labels == s)
            if len(arg_where) != 0:
                labeled_tp = data.T[arg_where][:, 0, :].T
                labeled_gfp = gfp[arg_where][:, 0]
                state_array = np.array([state] * len(arg_where)).transpose()

                dist_corr = _corr_vectors(state_array, labeled_tp)
                d[f"{state_name}_mean_corr"] = np.mean(np.abs(dist_corr))
                dist_gev = (labeled_gfp * dist_corr) ** 2 / np.sum(gfp**2)
                d[f"{state_name}_gev"] = np.sum(dist_gev)

                s_segments = np.array(
                    [len(group) for s_, group in segments if s_ == s]
                )
                occurrences = (
                    len(s_segments) / len(np.where(labels != -1)[0]) * sfreq
                )
                d[f"{state_name}_occurrences"] = occurrences

                timecov = np.sum(s_segments) / len(np.where(labels != -1)[0])
                d[f"{state_name}_timecov"] = timecov

                dist_durs = s_segments / sfreq
                d[f"{state_name}_meandurs"] = np.mean(dist_durs)

                if return_dist:
                    d[f"{state_name}_dist_corr"] = dist_corr
                    d[f"{state_name}_dist_gev"] = dist_gev
                    d[f"{state_name}_dist_durs"] = dist_durs

            else:
                d[f"{state_name}_mean_corr"] = 0
                d[f"{state_name}_gev"] = 0
                d[f"{state_name}_timecov"] = 0
                d[f"{state_name}_meandurs"] = 0
                d[f"{state_name}_occurrences"] = 0

                if return_dist:
                    d[f"{state_name}_dist_corr"] = np.array([])
                    d[f"{state_name}_dist_gev"] = np.array([])
                    d[f"{state_name}_dist_durs"] = np.array([])

        d["unlabeled"] = len(np.argwhere(labels == -1)) / len(gfp)
        return d

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
    def labels(self) -> NDArray[int]:
        """Microstate label attributed to each sample (the segmentation).

        :type: `~numpy.array`
        """
        return self._labels.copy()

    @property
    def cluster_centers_(self) -> NDArray[float]:
        """Fitted clusters (the mirostates maps).

        :type: `~numpy.array`
        """
        return self._cluster_centers_.copy()

    @property
    def cluster_names(self) -> List[str]:
        """Name of the clusters.

        :type: `list`
        """
        return self._cluster_names.copy()


@fill_doc
class RawSegmentation(_BaseSegmentation):
    """
    Contains the segmentation for a `~mne.io.Raw` instance.

    Parameters
    ----------
    %(labels_raw)s
    raw : Raw
        `~mne.io.Raw` instance used for prediction.
    %(cluster_centers)s
    %(cluster_names)s
    %(predict_parameters)s
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _check_type(self._inst, (BaseRaw,), item_name="raw")
        if self._labels.ndim != 1:
            raise ValueError(
                "Argument 'labels' should be a 1D array. The provided array "
                f"shape is {self._labels.shape} which has {self._labels.ndim} "
                "dimensions."
            )

        if self._inst.times.size != self._labels.shape[-1]:
            raise ValueError(
                "Provided MNE raw and labels do not have the same number "
                f"of samples. The 'raw' has {self._inst.times.size} samples, "
                f"while the 'labels' has {self._labels.shape[-1]} samples."
            )

    @fill_doc
    def plot(
        self,
        tmin: Optional[Union[int, float]] = None,
        tmax: Optional[Union[int, float]] = None,
        cmap: Optional[str] = None,
        axes: Optional[Axes] = None,
        cbar_axes: Optional[Axes] = None,
        *,
        block: bool = False,
        verbose: Optional[str] = None,
    ):
        """Plot the segmentation.

        Parameters
        ----------
        %(tmin_raw)s
        %(tmax_raw)s
        %(cmap)s
        %(axes_seg)s
        %(axes_cbar)s
        %(block)s
        %(verbose)s

        Returns
        -------
        fig : Figure
            Matplotlib figure containing the segmentation.
        """
        # Error checking on the input is performed in the viz function.
        return plot_raw_segmentation(
            labels=self._labels,
            raw=self._inst,
            n_clusters=self._cluster_centers_.shape[0],
            cluster_names=self._cluster_names,
            tmin=tmin,
            tmax=tmax,
            cmap=cmap,
            axes=axes,
            cbar_axes=cbar_axes,
            block=block,
            verbose=verbose,
        )

    # --------------------------------------------------------------------
    @property
    def raw(self) -> BaseRaw:
        """Raw instance."""
        return self._inst.copy()


@fill_doc
class EpochsSegmentation(_BaseSegmentation):
    """Contains the segmentation for an epoch instance.

    Parameters
    ----------
    %(labels_epo)s
    epochs : Epochs
        `~mne.Epochs` instance used for prediction.
    %(cluster_centers)s
    %(cluster_names)s
    %(predict_parameters)s
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _check_type(self._inst, (BaseEpochs,), "epochs")

        if self._labels.ndim != 2:
            raise ValueError(
                "Argument 'labels' should be a 2D array. The provided array "
                f"shape is {self._labels.shape} which has {self._labels.ndim} "
                "dimensions."
            )
        if len(self._inst) != self._labels.shape[0]:
            raise ValueError(
                "Provided MNE instance and labels do not have the same number "
                f"of epochs. The 'MNE instance' has {len(self._inst)} epochs, "
                f"while the 'labels' has {self._labels.shape[0]} epochs."
            )
        if self._inst.times.size != self._labels.shape[-1]:
            raise ValueError(
                "Provided MNE epochs and labels do not have the same number "
                f"of samples. The 'epochs' have {self._inst.times.size} "
                f"samples, while the 'labels' has {self._labels.shape[-1]} "
                "samples."
            )

    @fill_doc
    def plot(
        self,
        cmap: Optional[str] = None,
        axes: Optional[Axes] = None,
        cbar_axes: Optional[Axes] = None,
        *,
        block: bool = False,
        verbose=None,
    ):
        """Plot segmentation.

        Parameters
        ----------
        %(cmap)s
        %(axes_seg)s
        %(axes_cbar)s
        %(block)s
        %(verbose)s

        Returns
        -------
        fig : Figure
            Matplotlib figure containing the segmentation.
        """
        # Error checking on the input is performed in the viz function.
        return plot_epoch_segmentation(
            labels=self._labels,
            epochs=self._inst,
            n_clusters=self._cluster_centers_.shape[0],
            cluster_names=self._cluster_names,
            cmap=cmap,
            axes=axes,
            cbar_axes=cbar_axes,
            block=block,
            verbose=verbose,
        )

    # --------------------------------------------------------------------
    @property
    def epochs(self) -> BaseEpochs:
        """Epochs instance."""
        return self._inst.copy()
