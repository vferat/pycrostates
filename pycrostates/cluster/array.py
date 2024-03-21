from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from mne import Info

from ..io import ChInfo
from ..utils._checks import _check_type
from ..utils._docs import copy_doc
from ._base import _BaseCluster

if TYPE_CHECKING:
    from typing import Optional, Union

    from numpy.typing import NDArray


class ClusterArray(_BaseCluster):
    def __init__(
        self,
        data: NDArray[float],
        info: Union[Info, ChInfo],
        cluster_names: Optional[Union[list[str], tuple[str]]] = None,
        fitted_data: Optional[NDArray[float]] = None,
        labels: Optional[NDArray[int]] = None,
        ignore_polarity: bool = True,
    ) -> None:
        if not ignore_polarity:
            raise NotImplementedError(
                "pycrostates does not yet support 'ignore_polarity=False'."
            )
        super().__init__()
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
                f"number of channels in 'info' ({len(info["ch_names"])})."
            )
        self._n_clusters = data.shape[0]
        self._info = ChInfo(info=info)  # no-op if a ChInfo is priovided
        self._cluster_centers_ = data
        # validate optional inputs
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
        # validate fitted_data, which is either from a Raw ()

    # left for empty
    @copy_doc(_BaseCluster.save)
    def save(self, fname: Union[str, Path]):
        super().save(fname)
        from ..io.fiff import _write_cluster  # pylint: disable=C0415

        _write_cluster(
            fname,
            self._cluster_centers_,
            self._info,
            "array",
            self._cluster_names,
            self._fitted_data,
            self._labels_,
            GEV_=self._GEV_,
        )

    # # --------------------------------------------------------------------
    # @property
    # def GEV_(self) -> float:
    #     """Global Explained Variance.

    #     :type: `float`
    #     """
    #     if self._GEV_ is None:
    #         assert not self._fitted  # sanity-check
    #         logger.warning("Clustering algorithm has not been fitted.")
    #     return self._GEV_

    #     # set cluster_centers
    #     self._cluster_centers_ = maps
    #     # set sample_labels
    #     self._labels_ = segmentation

    #     # calculate gev
    #     gfp_sum_sq = np.sum(data**2)
    #     tmp_cluster_center = self.cluster_centers_
    #     tmp_label = self.labels_
    #     map_corr = _corr_vectors(data, tmp_cluster_center[tmp_label].T)
    #     gev = np.sum((data * map_corr) ** 2) / gfp_sum_sq
    #     self._GEV_ = gev

    #     # set fitted
    #     self._fitted = True

    #     return None
