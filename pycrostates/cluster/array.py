from __future__ import annotations  # c.f. PEP 563, PEP 649

from pathlib import Path
from typing import TYPE_CHECKING
from warnings import warn

import numpy as np
from mne import Info

from ..utils._checks import _check_type
from ..utils._docs import copy_doc
from ._base import _BaseCluster

if TYPE_CHECKING:
    from typing import Optional, Union

    from .._typing import ScalarFloatArray, ScalarIntArray
    from ..io import ChInfo


class ClusterArray(_BaseCluster):
    def __init__(
        self,
        data: ScalarFloatArray,
        info: Union[Info, ChInfo],
        cluster_names: Optional[Union[list[str], tuple[str]]] = None,
        fitted_data: Optional[ScalarFloatArray] = None,
        labels: Optional[ScalarIntArray] = None,
        ignore_polarity: bool = True,
    ) -> None:
        from ..io import ChInfo

        if not ignore_polarity:
            raise NotImplementedError(
                "pycrostates does not yet support 'ignore_polarity=False'."
            )
        super().__init__()
        self._ignore_polarity = ignore_polarity
        self._fitted = True
        # validate data and info
        _check_type(data, (np.ndarray,), "data")
        if data.ndim != 2:
            raise ValueError(
                "'data', the cluster centers, must be a 2D array or shape (n_clusters, "
                f"n_channels). Provided {data.ndim}D array is invalid."
            )
        _check_type(info, (Info, ChInfo), "info")
        ch_types = info.get_channel_types(unique=True)
        if len(ch_types):
            raise ValueError(
                "The 'info' object must not contain multiple channel types. "
                "The provided info has the following channel types: "
                f"{', '.join(ch_types)}"
            )
        if len(info["ch_names"]) != data.shape[1]:
            raise ValueError(
                f"The number of channels in 'data' ({data.shape[1]}) must match the "
                f"number of channels in 'info' ({len(info['ch_names'])})."
            )
        self._n_clusters = data.shape[0]
        self._info = ChInfo(info=info)  # no-op if a ChInfo is priovided
        self._cluster_centers_ = data
        # validate cluster names
        if cluster_names is not None:
            _check_type(cluster_names, (list, tuple), "cluster_names")
            if len(cluster_names) != self._n_clusters:
                raise ValueError(
                    f"The number of cluster names ({len(cluster_names)}) must match "
                    f"the number of clusters ({self._n_clusters})."
                )
            self._cluster_names = cluster_names
        else:
            self._cluster_names = [str(k) for k in range(self._n_clusters)]
        # validate fitted_data, which is either from a Raw (n_channels, n_times) or
        # Epochs (n_epochs, n_channels, n_times).
        _check_type(fitted_data, (np.ndarray, None), "fitted_data")
        if fitted_data is not None and fitted_data.ndim == 2:
            if fitted_data.shape[0] != len(info["ch_names"]):
                raise ValueError(
                    f"The number of channels in 'fitted_data' ({fitted_data.shape[0]}) "
                    "must match the number of channels in 'info' "
                    f"({len(info['ch_names'])})."
                )
        elif fitted_data is not None and fitted_data.ndim == 3:
            # either with the (n_channels,) as first or second dimension
            if fitted_data.shape[1] != len(info["ch_names"]):
                raise ValueError(
                    "The number of channels in 'fitted_data' "
                    f"({fitted_data.shape[1]}) must match the number of channels "
                    f"in 'info' ({len(info['ch_names'])}). Please provide "
                    "'fitted_data' as (n_epochs, n_channels, n_times) for Epochs."
                )
            fitted_data = np.swapaxes(fitted_data, 0, 1)
            fitted_data = fitted_data.reshape(fitted_data.shape[0], -1)
        else:
            raise ValueError(
                "'fitted_data' must be a 2D (raw, ChData) or 3D (epochs) array. The "
                f"provided {fitted_data.ndim}D array is invalid."
            )
        self._fitted_data = fitted_data
        # validate labels
        _check_type(labels, (np.ndarray, None), "labels")
        if labels is not None:
            if labels.ndim != 1:
                raise ValueError(
                    f"'labels' must be a 1D array. The provided {labels.ndim}D array "
                    "is invalid."
                )
            if self._fitted_data is None:
                warn(
                    RuntimeWarning,
                    "'labels' were provided without 'fitted_data' to which "
                    "they apply.",
                    stacklevel=2,
                )
            else:
                if labels.size != self._fitted_data.shape[1]:
                    raise ValueError(
                        f"The number of samples in 'labels' ({labels.size}) must match "
                        "the number of samples in 'fitted_data' "
                        f"({self._fitted_data.shape[1]})."
                    )
        self._labels_ = labels
        # estimate GEV
        if self._fitted_data is not None and self._labels_ is not None:
            self._gev = ClusterArray._compute_gev(self._fitted_data, self._labels_)

    @copy_doc(_BaseCluster.save)
    def save(self, fname: Union[str, Path]):
        super().save(fname)
        from ..io.fiff import _write_cluster  # pylint: disable=C0415

        _write_cluster(
            fname,
            self._cluster_centers_,
            self._info,
            "Array",
            self._cluster_names,
            self._fitted_data,
            self._labels_,
            ignore_polarity=self._ignore_polarity,
        )

    @staticmethod
    def _compute_gev(
        data: ScalarFloatArray,
        labels: ScalarIntArray,
        cluster_centers: ScalarFloatArray,
    ) -> float:
        """Compute Global Explained Variance (GEV).

        The GEV is the ratio of the variance explained by the clustering to the total
        variance of the data.

        Parameters
        ----------
        data : array of shape (n_channels, n_samples)
            The data to cluster.
        labels : array of shape (n_samples,)
            The cluster labels.
        cluster_centers : array of shape (n_clusters, n_channels)
            The cluster centers.

        Returns
        -------
        gev : float
            The Global Explained Variance.
        """
        pass
