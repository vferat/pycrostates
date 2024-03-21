from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from mne import BaseEpochs
from mne.io import BaseRaw
from numpy.typing import NDArray

from .._typing import Picks
from ..utils import _corr_vectors
from ..utils._docs import copy_doc
from ..utils._logs import logger
from ._base import _BaseCluster


class Custom(_BaseCluster):
    def __init__(self, n_clusters: int):
        super().__init__()

        # needed codes for custom
        self._n_clusters = _BaseCluster._check_n_clusters(n_clusters)
        self._cluster_names = [str(k) for k in range(self.n_clusters)]

        # fit variables
        self._GEV_ = None

    # left for empty
    def _repr_html_(self, caption=None):
        return None

    # left for empty
    @copy_doc(_BaseCluster.__eq__)
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Custom):
            if not super().__eq__(other):
                return False

            attributes = ("_GEV_",)
            for attribute in attributes:
                try:
                    attr1 = self.__getattribute__(attribute)
                    attr2 = other.__getattribute__(attribute)
                except AttributeError:
                    return False
                if attr1 != attr2:
                    return False

            return True
        return None

    # left for empty
    @copy_doc(_BaseCluster.__ne__)
    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def _check_fit(self):
        super()._check_fit()
        # sanity-check
        assert self.GEV_ is not None

    #
    def fit(
        self,
        inst: Union[BaseRaw, BaseEpochs],
        maps: NDArray,
        segmentation: NDArray,
        picks: Picks = "eeg",
        tmin: Optional[Union[int, float]] = None,
        tmax: Optional[Union[int, float]] = None,
        reject_by_annotation: bool = True,
        *,
        verbose: Optional[str] = None,
    ) -> None:
        data = super().fit(
            inst,
            picks=picks,
            tmin=tmin,
            tmax=tmax,
            reject_by_annotation=reject_by_annotation,
            verbose=verbose,
        )

        # set cluster_centers
        self._cluster_centers_ = maps
        # set sample_labels
        self._labels_ = segmentation

        # calculate gev
        gfp_sum_sq = np.sum(data**2)
        tmp_cluster_center = self.cluster_centers_
        tmp_label = self.labels_
        map_corr = _corr_vectors(data, tmp_cluster_center[tmp_label].T)
        gev = np.sum((data * map_corr) ** 2) / gfp_sum_sq
        self._GEV_ = gev

        # set fitted
        self._fitted = True

        return None

    # left for empty
    @copy_doc(_BaseCluster.save)
    def save(self, fname: Union[str, Path]):
        super().save(fname)
        # TODO: to be replaced by a general writer than infers the writer from
        # the file extension.
        # pylint: disable=import-outside-toplevel
        from ..io.fiff import _write_cluster

        # pylint: enable=import-outside-toplevel

        _write_cluster(
            fname,
            self._cluster_centers_,
            self._info,
            "Custom",
            self._cluster_names,
            self._fitted_data,
            self._labels_,
            GEV_=self._GEV_,
        )

    # --------------------------------------------------------------------
    # No static method

    # --------------------------------------------------------------------
    @property
    def GEV_(self) -> float:
        """Global Explained Variance.

        :type: `float`
        """
        if self._GEV_ is None:
            assert not self._fitted  # sanity-check
            logger.warning("Clustering algorithm has not been fitted.")
        return self._GEV_

    @_BaseCluster.fitted.setter
    @copy_doc(_BaseCluster.fitted.setter)
    def fitted(self, fitted):
        super(self.__class__, self.__class__).fitted.__set__(self, fitted)
        if not fitted:
            self._GEV_ = None
