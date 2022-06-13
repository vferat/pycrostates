from abc import ABC, abstractmethod
from copy import copy, deepcopy
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from matplotlib.axes import Axes
from mne import BaseEpochs, pick_info
from mne.annotations import _annotations_starts_stops
from mne.io import BaseRaw
from mne.io.pick import _picks_to_idx
from numpy.typing import NDArray
from scipy.signal import convolve2d

from .._typing import CHData, Picks
from ..segmentation import EpochsSegmentation, RawSegmentation
from ..utils import _compare_infos, _corr_vectors
from ..utils._checks import (
    _check_n_jobs,
    _check_reject_by_annotation,
    _check_tmin_tmax,
    _check_type,
    _check_value,
)
from ..utils._docs import fill_doc
from ..utils._logs import logger, verbose
from ..utils.mixin import ChannelsMixin, ContainsMixin, MontageMixin
from ..viz import plot_cluster_centers


class _BaseCluster(ABC, ChannelsMixin, ContainsMixin, MontageMixin):
    """Base Class for Microstates Clustering algorithms."""

    @abstractmethod
    def __init__(self):
        self._n_clusters = None
        self._cluster_names = None
        self._cluster_centers_ = None

        # fit variables
        self._info = None
        self._fitted_data = None
        self._labels_ = None
        self._fitted = False

    def __repr__(self) -> str:  # noqa: D401
        """String representation."""
        name = self.__class__.__name__
        if self.fitted:
            s = f"<{name} | fitted on n = {self.n_clusters} cluster centers>"
        else:
            s = f"<{name} | not fitted>"
        return s

    def _repr_html_(self, caption=None):
        """HTML representation."""
        from ..html_templates import repr_templates_env

        template = repr_templates_env.get_template("BaseCluster.html.jinja")
        if self.fitted:
            n_samples = self._fitted_data.shape[-1]
            ch_types, ch_counts = np.unique(
                self.get_channel_types(), return_counts=True
            )
            ch_repr = [
                f"{ch_count} {ch_type.upper()}"
                for ch_type, ch_count in zip(ch_types, ch_counts)
            ]
        else:
            n_samples = None
            ch_repr = None

        return template.render(
            name=self.__class__.__name__,
            n_clusters=self._n_clusters,
            cluster_names=self._cluster_names,
            fitted=self._fitted,
            n_samples=n_samples,
            ch_repr=ch_repr,
        )

    def __eq__(self, other: Any) -> bool:
        """Equality == method."""
        if isinstance(other, _BaseCluster):
            # check fit
            if self._fitted + other._fitted == 0:  # Both False
                raise RuntimeError(
                    "Clustering algorithms must be fitted before using '==' "
                    "comparison."
                )
            if self._fitted + other._fitted == 1:  # One False
                return False

            attributes = (
                "_n_clusters",
                "_info",
            )

            for attribute in attributes:
                try:
                    attr1 = self.__getattribute__(attribute)
                    attr2 = other.__getattribute__(attribute)
                except AttributeError:
                    return False
                if attr1 != attr2:
                    return False

            array_attributes = (
                "_cluster_centers_",
                "_fitted_data",
                "_labels_",
            )
            for attribute in array_attributes:
                try:
                    attr1 = self.__getattribute__(attribute)
                    attr2 = other.__getattribute__(attribute)
                except AttributeError:
                    return False
                if attr1.shape != attr2.shape:
                    return False
                if not np.allclose(attr1, attr2):
                    return False

            # check cluster names
            assert len(self._cluster_names) == self._n_clusters
            assert len(other._cluster_names) == other._n_clusters
            if self._cluster_names != other._cluster_names:
                logger.warning(
                    "Cluster names differ between both clustering solution. "
                    "Consider using '.rename_clusters' to change the cluster "
                    "names."
                )

            return True
        else:
            return False

    def __ne__(self, other: Any) -> bool:
        """Different != method."""
        return not self.__eq__(other)

    def copy(
        self,
        deep: bool = True,
    ):
        """Return a copy of the instance.

        Parameters
        ----------
        deep : bool
            If True, `~copy.deepcopy` is used instead of `~copy.copy`.
        """
        if deep:
            return deepcopy(self)
        return copy(self)

    def _check_fit(self):
        """Check if the cluster is fitted."""
        if not self.fitted:
            raise RuntimeError(
                "Clustering algorithm must be fitted before using "
                f"{self.__class__.__name__}"
            )
        # sanity-check
        assert self._cluster_centers_ is not None
        assert self._info is not None
        assert self._fitted_data is not None
        assert self._labels_ is not None

    @abstractmethod
    @fill_doc
    def fit(
        self,
        inst: Union[BaseRaw, BaseEpochs, CHData],
        picks: Picks = None,
        tmin: Optional[Union[int, float]] = None,
        tmax: Optional[Union[int, float]] = None,
        reject_by_annotation: bool = True,
        n_jobs: int = 1,
    ) -> NDArray[float]:
        """Segment `~mne.io.Raw` or `~mne.Epochs` into microstate sequence.

        Parameters
        ----------
        inst : Raw | Epochs
            MNE `~mne.io.Raw` or `~mne.Epochs` object containing data to
            transform to cluster-distance space (absolute spatial correlation).
        %(picks_all)s
        %(tmin_raw)s
        %(tmax_raw)s
        %(reject_by_annotation_raw)s
        %(n_jobs)s
        """
        from ..io import ChData, ChInfo

        n_jobs = _check_n_jobs(n_jobs)
        _check_type(inst, (BaseRaw, BaseEpochs, ChData), item_name="inst")
        if isinstance(inst, (BaseRaw, BaseEpochs)):
            tmin, tmax = _check_tmin_tmax(inst, tmin, tmax)
        if isinstance(inst, BaseRaw):
            reject_by_annotation = _check_reject_by_annotation(
                reject_by_annotation
            )

        # retrieve info
        info = inst.info

        # picks
        picks_bads_inc = _picks_to_idx(info, picks, none="all", exclude=[])
        picks = _picks_to_idx(info, picks, none="all", exclude="bads")
        ch_not_used = set(picks_bads_inc) - set(picks)
        if len(ch_not_used) != 0:
            if len(ch_not_used) == 1:
                msg = (
                    "Channel %s is set as bad and ignored. To include "
                    + "it, either remove it from 'inst.info['bads'] or "
                    + "provide its name explicitly in the 'picks' argument."
                )
            else:
                msg = (
                    "Channels %s are set as bads and ignored. To "
                    + "include them, either remove them from "
                    + "'inst.info['bads'] or provide their names "
                    + "explicitly in the 'picks' argument."
                )
            logger.warning(
                msg, ", ".join(info["ch_names"][k] for k in ch_not_used)
            )
            del msg

        # retrieve numpy array
        kwargs = (
            dict() if isinstance(inst, ChData) else dict(tmin=tmin, tmax=tmax)
        )
        if isinstance(inst, BaseRaw):
            kwargs["reject_by_annotation"] = reject_by_annotation
        data = inst.get_data(picks=picks, **kwargs)
        # reshape if inst is Epochs
        if isinstance(inst, BaseEpochs):
            data = np.swapaxes(data, 0, 1)
            data = data.reshape(data.shape[0], -1)

        # store picks and info
        self._info = ChInfo(info=pick_info(info, picks_bads_inc))
        self._fitted_data = data

        return data

    def rename_clusters(
        self,
        mapping: Optional[Dict[str, str]] = None,
        new_names: Optional[
            Union[
                List[str],
                Tuple[str, ...],
            ]
        ] = None,
    ) -> None:
        """Rename the clusters.

        Parameters
        ----------
        mapping : dict
            Mapping from the old names to the new names. The keys are the old
            names and the values are the new names.
        new_names : list | tuple
            1D iterable containing the new cluster names. The length of the
            iterable should match the number of clusters.

        Notes
        -----
        Operates in-place.
        """
        self._check_fit()

        if mapping is not None and new_names is not None:
            raise ValueError(
                "Only one of 'mapping' or 'new_names' must be provided."
            )

        if mapping is not None:
            _check_type(mapping, (dict,), item_name="mapping")
            for key in mapping:
                _check_value(key, self._cluster_names, item_name="old name")
            for value in mapping.values():
                _check_type(value, (str,), item_name="new name")

        elif new_names is not None:
            _check_type(new_names, (list, tuple), item_name="new_names")
            if len(new_names) != self._n_clusters:
                raise ValueError(
                    "Argument 'new_names' should contain 'n_clusters': "
                    f"{self._n_clusters} elements. "
                    f"Provided '{len(new_names)}'."
                )

            # sanity-check
            assert len(self._cluster_names) == len(new_names)

            # convert to dict
            mapping = {
                old_name: new_names[k]
                for k, old_name in enumerate(self._cluster_names)
            }

        else:
            logger.warning(
                "Either 'mapping' or 'new_names' should not be 'None' "
                "for method 'rename_clusters' to operate."
            )
            return

        self._cluster_names = [
            mapping[name] if name in mapping else name
            for name in self._cluster_names
        ]

    def reorder_clusters(
        self,
        mapping: Optional[Dict[int, int]] = None,
        order: Optional[
            Union[
                List[int],
                Tuple[int, ...],
                NDArray[int],
            ]
        ] = None,
    ) -> None:
        """Reorder the clusters.

        Parameters
        ----------
        mapping : dict
            Mapping from the old order to the new order.
            key: old position, value: new position.
        order : list of int | tuple of int | array of int
            1D iterable containing the new order.

        Notes
        -----
        The positions are 0-indexed.
        Operates in-place.
        """
        self._check_fit()

        if mapping is not None and order is not None:
            raise ValueError(
                "Only one of 'mapping' or 'order' must be provided."
            )

        if mapping is not None:
            _check_type(mapping, (dict,), item_name="mapping")
            valids = tuple(range(self._n_clusters))
            for key in mapping:
                _check_value(key, valids, item_name="old position")
            for value in mapping.values():
                _check_value(value, valids, item_name="new position")

            inverse_mapping = {value: key for key, value in mapping.items()}

            # check uniqueness
            if len(set(mapping.values())) != len(mapping.values()):
                raise ValueError(
                    "Position in the new order can not be repeated."
                )
            # check that a cluster is not moved twice
            for key in mapping:
                if key in mapping.values():
                    raise ValueError(
                        "A position can not be present in both the old and "
                        f"new order. Position '{key}' is mapped to "
                        f"'{mapping[key]}' and position "
                        f"'{inverse_mapping[key]}' is mapped to '{key}'."
                    )

            # convert to list
            order = list(range(self._n_clusters))
            for key, value in mapping.items():
                order[key] = value
                order[value] = key
            # sanity-check
            assert len(set(order)) == self._n_clusters

        elif order is not None:
            _check_type(order, (list, tuple, np.ndarray), item_name="order")
            if isinstance(order, np.ndarray) and len(order.shape) != 1:
                raise ValueError(
                    "Argument 'order' should be a 1D iterable and not a "
                    f"{len(order.shape)}D iterable."
                )
            valids = tuple(range(self._n_clusters))
            for elt in order:
                _check_value(elt, valids, item_name="order")
            if len(order) != self._n_clusters:
                raise ValueError(
                    "Argument 'order' should contain 'n_clusters': "
                    f"{self._n_clusters} elements. Provided '{len(order)}'."
                )
            order = list(order)

        else:
            logger.warning(
                "Either 'mapping' or 'order' should not be 'None' "
                "for method 'reorder_clusters' to operate."
            )
            return

        # re-order
        self._cluster_centers_ = self._cluster_centers_[order]
        self._cluster_names = [self._cluster_names[k] for k in order]
        new_labels = np.full(self._labels_.shape, -1)
        for k in range(0, self.n_clusters):
            new_labels[self._labels_ == k] = order[k]
        self._labels_ = new_labels

    def invert_polarity(
        self,
        invert: Union[
            bool,
            List[bool],
            Tuple[bool, ...],
            NDArray[bool],
        ],
    ) -> None:
        """Invert map polarities for visualisation purposes.

        Parameters
        ----------
        invert : bool | list of bool | array of bool
            List of bool of length ``n_clusters``.
            True will invert map polarity, while False will have no effect.
            If a bool is provided, it is applied to all maps.

        Notes
        -----
        Operates in-place.
        """
        self._check_fit()

        # Check argument
        invert = _check_type(
            invert, (bool, list, tuple, np.ndarray), item_name="invert"
        )
        if isinstance(invert, bool):
            invert = [invert] * self._n_clusters
        elif isinstance(invert, (list, tuple)):
            for inv in invert:
                _check_type(inv, (bool,), item_name="invert")
        elif isinstance(invert, np.ndarray):
            if len(invert.shape) != 1:
                raise ValueError(
                    "Argument 'invert' should be a 1D iterable and not a "
                    f"{len(invert.shape)}D iterable."
                )
            for inv in invert:
                _check_type(inv, (bool, np.bool_), item_name="invert")
        if len(invert) != self._n_clusters:
            raise ValueError(
                "Argument 'invert' should be either a bool or a list of bools "
                f"of length 'n_clusters' ({self._n_clusters}). The provided "
                f"'invert' length is '{len(invert)}'."
            )

        # Invert maps
        for k, cluster in enumerate(self._cluster_centers_):
            if invert[k]:
                self._cluster_centers_[k] = -cluster

    def plot(self, axes: Optional[Axes] = None, *, block: bool = False):
        """
        Plot cluster centers as topographic maps.

        Parameters
        ----------
        axes : Axes | None
            Either none to create a new figure or axes (or an array of axes)
            on which the topographic map should be plotted.
        block : bool
            Whether to halt program execution until the figure is closed.

        Returns
        -------
        f : Figure
            Matplotlib figure containing the topographic plots.
        """
        self._check_fit()
        picks = _picks_to_idx(self._info, "all", none="all", exclude="bads")
        info = pick_info(self._info, picks)
        return plot_cluster_centers(
            self._cluster_centers_,
            info,
            self._cluster_names,
            axes,
            block=block,
        )

    @abstractmethod
    def save(self, fname: Union[str, Path]):
        """
        Save clustering solution to disk.

        Parameters
        ----------
        fname : path-like
            Path to the ``.fif`` file where the clustering solution is saved.
        """
        self._check_fit()
        _check_type(fname, ("path-like",), "fname")

    @verbose
    def predict(
        self,
        inst: Union[BaseRaw, BaseEpochs],
        picks: Picks = None,
        factor: int = 0,
        half_window_size: int = 3,
        tol: Union[int, float] = 10e-6,
        min_segment_length: int = 0,
        reject_edges: bool = True,
        reject_by_annotation: bool = True,
        *,
        verbose: Optional[str] = None,
    ):
        """Segment `~mne.io.Raw` or `~mne.Epochs` into microstate sequence.

        Parameters
        ----------
        inst : Raw | Epochs
            MNE `~mne.io.Raw` or `~mne.Epochs` object containing the data to
            use for prediction.
        %(picks_all)s
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
        segmentation : `RawSegmentation` | `EpochsSegmentation`
            Microstate sequence derivated from instance data. Timepoints are
            labeled according to cluster centers number: 0 for the first
            center, 1 for the second, etc..
            -1 is used for unlabeled time points.
        """
        # TODO: reject_by_annotation_raw doc probably doesn't match the correct
        # argument types.
        self._check_fit()
        _check_type(inst, (BaseRaw, BaseEpochs), item_name="inst")
        _check_type(factor, ("int",), item_name="factor")
        _check_type(half_window_size, ("int",), item_name="half_window_size")
        _check_type(tol, ("numeric",), item_name="tol")
        _check_type(
            min_segment_length, ("int",), item_name="min_segment_length"
        )
        _check_type(reject_edges, (bool,), item_name="reject_edges")
        _check_type(
            reject_by_annotation,
            (bool, str, None),
            item_name="reject_by_annotation",
        )
        if isinstance(reject_by_annotation, str):
            if reject_by_annotation == "omit":
                reject_by_annotation = True
            else:
                raise ValueError(
                    "Argument 'reject_by_annotation' can be set to 'True', "
                    f"'False' or 'omit' (True). '{reject_by_annotation}' is "
                    "not supported."
                )
        elif reject_by_annotation is None:
            reject_by_annotation = False

        # check that the instance as the required channels (good + bads)
        # inst_info must have all the channels present in cluster_info
        _compare_infos(cluster_info=self._info, inst_info=inst.info)
        picks_ = _picks_to_idx(inst.info, picks, none="all", exclude="bads")
        ch_ = [
            ch
            for k, ch in enumerate(inst.info["ch_names"])
            if k in picks_ and ch in self._info["bads"]
        ]
        if 1 == len(ch_):
            logger.warning(
                "Picked channel %s was set as "
                "bads during fitting and will be ignored.",
                ch_[0],
            )
        elif 1 < len(ch_):
            logger.warning(
                "Picked channels %s were set as "
                "bads during fitting and will be ignored.",
                ", ".join(ch_),
            )

        # remove channels that were bads during fitting from picks
        picks_ = [
            ch
            for k, ch in enumerate(inst.info["ch_names"])
            if k in picks_ and ch not in self._info["bads"]
        ]
        picks_data = _picks_to_idx(inst.info, picks_, none="all", exclude=[])
        good_channels = [
            ch for ch in self._info["ch_names"] if ch not in self._info["bads"]
        ]
        picks_cluster_centers = np.array(
            [good_channels.index(ch) for ch in picks_]
        )

        # logging messages
        if factor == 0:
            logger.info("Segmenting data without smoothing.")
        else:
            logger.info(
                "Segmenting data with factor %s and effective smoothing "
                "window size: %.4f (ms).",
                factor,
                (2 * half_window_size + 1) / inst.info["sfreq"],
            )
        if min_segment_length > 0:
            logger.info(
                "Rejecting segments shorter than %.4f (ms).",
                min_segment_length / inst.info["sfreq"],
            )
        if reject_edges:
            logger.info("Rejecting first and last segments.")

        if isinstance(inst, BaseRaw):
            segmentation = self._predict_raw(
                inst,
                picks_data,
                picks_cluster_centers,
                factor,
                tol,
                half_window_size,
                min_segment_length,
                reject_edges,
                reject_by_annotation,
            )
        elif isinstance(inst, BaseEpochs):
            segmentation = self._predict_epochs(
                inst,
                picks_data,
                picks_cluster_centers,
                factor,
                tol,
                half_window_size,
                min_segment_length,
                reject_edges,
            )
        return segmentation

    def _predict_raw(
        self,
        raw: BaseRaw,
        picks_data: NDArray[int],
        picks_cluster_centers: NDArray[int],
        factor: int,
        tol: Union[int, float],
        half_window_size: int,
        min_segment_length: int,
        reject_edges: bool,
        reject_by_annotation: bool,
    ) -> RawSegmentation:
        """Create segmentation for raw."""
        predict_parameters = {
            "factor": factor,
            "tol": tol,
            "half_window_size": half_window_size,
            "min_segment_length": min_segment_length,
            "reject_edges": reject_edges,
            "reject_by_annotation": reject_by_annotation,
        }

        # retrieve data for picks
        data = raw.get_data(picks=picks_data)
        # retrieve cluster_centers_ for picks
        cluster_centers_ = deepcopy(self._cluster_centers_)
        cluster_centers_ = cluster_centers_[:, picks_cluster_centers]

        if reject_by_annotation:
            # retrieve onsets/ends for BAD annotations
            onsets, ends = _annotations_starts_stops(raw, ["BAD"])
            onsets = onsets.tolist() + [data.shape[-1] - 1]
            ends = [0] + ends.tolist()

            segmentation = np.full(data.shape[-1], -1)

            for onset, end in zip(onsets, ends):
                # small segments can't be smoothed
                if factor != 0 and onset - end < 2 * half_window_size + 1:
                    continue

                data_ = data[:, end:onset]
                segment = _BaseCluster._segment(
                    data_, cluster_centers_, factor, tol, half_window_size
                )
                if reject_edges:
                    segment = _BaseCluster._reject_edge_segments(segment)
                segmentation[end:onset] = segment

        else:
            segmentation = _BaseCluster._segment(
                data, cluster_centers_, factor, tol, half_window_size
            )
            if reject_edges:
                segmentation = _BaseCluster._reject_edge_segments(segmentation)

        if 0 < min_segment_length:
            segmentation = _BaseCluster._reject_short_segments(
                segmentation, data, min_segment_length
            )

        # Provide properties to copy the arrays
        return RawSegmentation(
            labels=segmentation,
            inst=raw.copy().pick(picks_data),
            cluster_centers_=self.cluster_centers_,
            cluster_names=self.cluster_names,
            predict_parameters=predict_parameters,
        )

    def _predict_epochs(
        self,
        epochs: BaseEpochs,
        picks_data: NDArray[int],
        picks_cluster_centers: NDArray[int],
        factor: int,
        tol: Union[int, float],
        half_window_size: int,
        min_segment_length: int,
        reject_edges: bool,
    ) -> EpochsSegmentation:
        """Create segmentation for epochs."""
        predict_parameters = {
            "factor": factor,
            "tol": tol,
            "half_window_size": half_window_size,
            "min_segment_length": min_segment_length,
            "reject_edges": reject_edges,
        }

        # retrieve data for picks
        data = epochs.get_data(picks=picks_data)
        # retrieve cluster_centers_ for picks
        cluster_centers_ = deepcopy(self._cluster_centers_)
        cluster_centers_ = cluster_centers_[:, picks_cluster_centers]

        segments = []
        for epoch_data in data:
            segment = _BaseCluster._segment(
                epoch_data, cluster_centers_, factor, tol, half_window_size
            )

            if 0 < min_segment_length:
                segment = _BaseCluster._reject_short_segments(
                    segment, epoch_data, min_segment_length
                )
            if reject_edges:
                segment = _BaseCluster._reject_edge_segments(segment)

            segments.append(segment)

        # Provide properties to copy the arrays
        return EpochsSegmentation(
            labels=np.array(segments),
            inst=epochs.copy().pick(picks_data),
            cluster_centers_=self.cluster_centers_,
            cluster_names=self.cluster_names,
            predict_parameters=predict_parameters,
        )

    # --------------------------------------------------------------------
    @staticmethod
    def _segment(
        data: NDArray[float],
        states: NDArray[float],
        factor: int,
        tol: Union[int, float],
        half_window_size: int,
    ) -> NDArray[int]:
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
                data, states, labels, factor, tol, half_window_size
            )

        return labels

    @staticmethod
    def _smooth_segmentation(
        data: NDArray[float],
        states: NDArray[float],
        labels: NDArray[int],
        factor: int,
        tol: Union[int, float],
        half_window_size: int,
    ) -> NDArray[int]:
        """Apply smoothing.

        Adapted from [1].

        References
        ----------
        .. [1] R. D. Pascual-Marqui, C. M. Michel and D. Lehmann.
            Segmentation of brain electrical activity into microstates:
            model estimation and validation.
            IEEE Transactions on Biomedical Engineering,
            vol. 42, no. 7, pp. 658-665, July 1995,
            https://doi.org/10.1109/10.391164.
        """
        Ne, Nt = data.shape
        Nu = states.shape[0]
        Vvar = np.sum(data * data, axis=0)
        rmat = np.tile(np.arange(0, Nu), (Nt, 1)).T

        w = np.zeros((Nu, Nt))
        w[(rmat == labels)] = 1
        e = np.sum(
            Vvar - np.sum(np.dot(w.T, states).T * data, axis=0) ** 2
        ) / (Nt * (Ne - 1))
        window = np.ones((1, 2 * half_window_size + 1))

        S0 = 0
        while True:
            Nb = convolve2d(w, window, mode="same")
            x = (np.tile(Vvar, (Nu, 1)) - (np.dot(states, data)) ** 2) / (
                2 * e * (Ne - 1)
            ) - factor * Nb
            dlt = np.argmin(x, axis=0)

            labels = dlt
            w = np.zeros((Nu, Nt))
            w[(rmat == labels)] = 1
            Su = np.sum(
                Vvar - np.sum(np.dot(w.T, states).T * data, axis=0) ** 2
            ) / (Nt * (Ne - 1))
            if np.abs(Su - S0) <= np.abs(tol * Su):
                break
            S0 = Su

        return labels

    @staticmethod
    def _reject_short_segments(
        segmentation: NDArray[int],
        data: NDArray[float],
        min_segment_length: int,
    ) -> NDArray[int]:
        """Reject segments that are too short.

        Reject segments that are too short by replacing the labels with the
        adjacent labels based on data correlation.
        """
        while True:
            # list all segments
            segments = [list(group) for _, group in groupby(segmentation)]
            idx = 0  # where does the segment start

            for k, segment in enumerate(segments):
                skip_condition = [
                    k in (0, len(segments) - 1),  # ignore edge segments
                    segment[0] == -1,  # ignore segments labelled with 0
                    min_segment_length
                    <= len(segment),  # ignore large segments
                ]
                if any(skip_condition):
                    idx += len(segment)
                    continue

                left = idx
                right = idx + len(segment) - 1
                new_segment = segmentation[left : right + 1]

                while len(new_segment) != 0:
                    # compute correlation left/right side
                    left_corr = np.abs(
                        _corr_vectors(
                            data[:, left - 1].T,
                            data[:, left].T,
                        )
                    )
                    right_corr = np.abs(
                        _corr_vectors(data[:, right].T, data[:, right + 1].T)
                    )

                    if np.abs(right_corr - left_corr) <= 1e-8:
                        # equal corr, try to do both sides
                        if len(new_segment) == 1:
                            # do only one side, left
                            segmentation[left] = segmentation[left - 1]
                            left += 1
                        else:
                            # If equal, do both sides
                            segmentation[right] = segmentation[right + 1]
                            segmentation[left] = segmentation[left - 1]
                            right -= 1
                            left += 1
                    else:
                        if left_corr < right_corr:
                            segmentation[right] = segmentation[right + 1]
                            right -= 1
                        elif right_corr < left_corr:
                            segmentation[left] = segmentation[left - 1]
                            left += 1

                    # crop segment
                    new_segment = segmentation[left : right + 1]

                # segments that were too short might have become long enough,
                # so list them again and check again.
                break
            else:
                break  # stop while loop because all segments are long enough

        return segmentation

    @staticmethod
    def _reject_edge_segments(segmentation: NDArray[int]) -> NDArray[int]:
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
    def n_clusters(self) -> int:
        """Number of clusters (number of microstates).

        :type: `int`
        """
        return self._n_clusters

    @property
    def info(self):
        """Info instance with the channel information used to fit the instance.

        :type: `~pycrostates.io.ChInfo`
        """
        if self._info is None:
            assert not self._fitted  # sanity-check
            logger.warning("Clustering algorithm has not been fitted.")
            return None
        return self._info

    @property
    def fitted(self) -> bool:
        """Fitted state.

        :type: `bool`
        """
        return self._fitted

    @fitted.setter
    def fitted(self, fitted):
        """Property-setter used to reset all fit variables."""
        _check_type(fitted, (bool,), item_name="fitted")
        if fitted and not self._fitted:
            logger.warning(
                "The property 'fitted' can not be set to 'True' directly. "
                "Please use the .fit() method to fit the clustering "
                "algorithm."
            )
        elif fitted and self._fitted:
            logger.warning(
                "The property 'fitted' can not be set to 'True' directly. "
                "The clustering algorithm has already been fitted."
            )
        else:
            self._cluster_centers_ = None
            self._info = None
            self._fitted_data = None
            self._labels_ = None
            self._fitted = False

    @property
    def cluster_centers_(self) -> NDArray[float]:
        """Fitted clusters (the microstates maps).

        Returns None if cluster algorithm has not been fitted.

        :type: `~numpy.array` (n_clusters, n_channels) | None
        """
        if self._cluster_centers_ is None:
            assert not self._fitted  # sanity-check
            logger.warning("Clustering algorithm has not been fitted.")
            return None
        return self._cluster_centers_.copy()

    @property
    def fitted_data(self) -> NDArray[float]:
        """Data array used to fit the clustering algorithm.

        :type: `~numpy.array` shape (n_channels, n_samples) | None
        """
        if self._fitted_data is None:
            assert not self._fitted  # sanity-check
            logger.warning("Clustering algorithm has not been fitted.")
            return None
        return self._fitted_data.copy()

    @property
    def labels_(self) -> NDArray[int]:
        """Microstate label attributed to each sample of the fitted data.

        :type: `~numpy.array` shape (n_samples, ) | None
        """
        if self._labels_ is None:
            assert not self._fitted  # sanity-check
            logger.warning("Clustering algorithm has not been fitted.")
            return None
        return self._labels_.copy()

    @property
    def cluster_names(self) -> List[str]:
        """Name of the clusters.

        :type: `list`
        """
        return self._cluster_names.copy()

    @cluster_names.setter
    def cluster_names(self, other: Any):
        """Setter for the cluster names."""
        logger.warning(
            "The attribute 'cluster_names' can not be set directly. Please "
            "use the 'rename_clusters' method instead."
        )

    # --------------------------------------------------------------------
    @staticmethod
    def _check_n_clusters(n_clusters: int) -> int:
        """Check that the number of clusters is a positive integer."""
        _check_type(n_clusters, ("int",), item_name="n_clusters")
        if n_clusters <= 0:
            raise ValueError(
                "The number of clusters must be a positive integer. "
                f"Provided: '{n_clusters}'."
            )
        return n_clusters
