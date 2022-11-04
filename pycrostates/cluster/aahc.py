"""Atomize and Agglomerate Hierarchical Clustering (AAHC)."""

from functools import wraps as func_wraps
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
from mne import BaseEpochs
from mne.io import BaseRaw
from numpy.typing import NDArray

from .._typing import Picks
from ..utils import _corr_vectors
from ..utils._checks import _check_type
from ..utils._docs import copy_doc, fill_doc
from ..utils._logs import logger
from ._base import _BaseCluster

# if we have numba, use its jit interface
try:
    from numba import njit
except ImportError:

    def njit(cache=False):  # pylint: disable=unused-argument
        """Define dummy decorator for numba."""

        def decorator(func):
            @func_wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator


@fill_doc
class AAHCluster(_BaseCluster):
    r"""Atomize and Agglomerate Hierarchical Clustering (AAHC) algorithm.

    See :footcite:t:`Murray2008` for additional information.

    Parameters
    ----------
    %(n_clusters)s
    normalize_input : bool
        If set, the input data is normalized along the channel dimension.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        n_clusters: int,
        normalize_input: bool = False,
    ):
        super().__init__()

        self._n_clusters = _BaseCluster._check_n_clusters(n_clusters)
        self._cluster_names = [str(k) for k in range(self.n_clusters)]

        # TODO : ignor_polarity=True for now.
        # After _BaseCluster and Metric support ignore_polarity
        # make the parameter an argument
        # https://github.com/vferat/pycrostates/pull/93#issue-1431122168
        self._ignore_polarity = True
        self._normalize_input = AAHCluster._check_normalize_input(
            normalize_input
        )

        # fit variables
        self._GEV_ = None

    def _repr_html_(self, caption=None):
        # pylint: disable=import-outside-toplevel
        from ..html_templates import repr_templates_env

        # pylint: enable=import-outside-toplevel

        template = repr_templates_env.get_template("AAHCluster.html.jinja")
        if self.fitted:
            n_samples = self._fitted_data.shape[-1]
            ch_types, ch_counts = np.unique(
                self.get_channel_types(), return_counts=True
            )
            ch_repr = [
                f"{ch_count} {ch_type.upper()}"
                for ch_type, ch_count in zip(ch_types, ch_counts)
            ]
            GEV = f"{self._GEV_ * 100:.2f}"
        else:
            n_samples = None
            ch_repr = None
            GEV = None

        return template.render(
            name=self.__class__.__name__,
            n_clusters=self._n_clusters,
            GEV=GEV,
            cluster_names=self._cluster_names,
            fitted=self._fitted,
            n_samples=n_samples,
            ch_repr=ch_repr,
        )

    @copy_doc(_BaseCluster.__eq__)
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, AAHCluster):
            if not super().__eq__(other):
                return False

            attributes = (
                "_ignore_polarity",
                "_normalize_input",
                "_GEV_",
            )
            for attribute in attributes:
                try:
                    attr1 = self.__getattribute__(attribute)
                    attr2 = other.__getattribute__(attribute)
                except AttributeError:
                    return False
                if attr1 != attr2:
                    return False

            return True
        return False

    @copy_doc(_BaseCluster.__ne__)
    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    @copy_doc(_BaseCluster._check_fit)
    def _check_fit(self):
        super()._check_fit()
        # sanity-check
        assert self.GEV_ is not None

    @copy_doc(_BaseCluster.fit)
    def fit(
        self,
        inst: Union[BaseRaw, BaseEpochs],
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

        gev, maps, segmentation = AAHCluster._aahc(
            data,
            self._n_clusters,
            self._ignore_polarity,
            self._normalize_input,
        )

        if gev is not None:
            logger.info("AAHC converged with GEV = %.2f%% ", gev * 100)

        self._GEV_ = gev
        self._cluster_centers_ = maps
        self._labels_ = segmentation
        self._fitted = True

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
            "AAHCluster",
            self._cluster_names,
            self._fitted_data,
            self._labels_,
            ignore_polarity=self._ignore_polarity,
            normalize_input=self._normalize_input,
            GEV_=self._GEV_,
        )

    # --------------------------------------------------------------------
    @staticmethod
    def _aahc(
        data: NDArray[float],
        n_clusters: int,
        ignore_polarity: bool,
        normalize_input: bool,
    ) -> Tuple[float, NDArray[float], NDArray[int]]:
        """Run the AAHC algorithm."""
        gfp_sum_sq = np.sum(data**2)
        maps, segmentation = AAHCluster._compute_maps(
            data, n_clusters, ignore_polarity, normalize_input
        )
        map_corr = _corr_vectors(data, maps[segmentation].T)
        gev = np.sum((data * map_corr) ** 2) / gfp_sum_sq
        return gev, maps, segmentation

    # pylint: disable=too-many-locals
    @staticmethod
    def _compute_maps(
        data: NDArray[float],
        n_clusters: int,
        ignore_polarity: bool,
        normalize_input: bool,
    ) -> Tuple[NDArray[float], NDArray[int]]:
        """Compute microstates maps."""
        n_chan, n_frame = data.shape

        cluster = data.copy()
        cluster /= np.linalg.norm(cluster, axis=0, keepdims=True)

        if normalize_input:
            data = cluster.copy()

        GEV = np.sum(data * cluster, axis=0)

        assignment = np.arange(n_frame)

        n_steps = n_frame - n_clusters
        while cluster.shape[1] > n_clusters:

            step = n_frame - cluster.shape[1]

            to_remove = np.argmin(GEV)
            orphans = assignment == to_remove

            old_cluster = cluster[:, to_remove].copy()

            cluster = np.delete(cluster, to_remove, axis=1)
            GEV = np.delete(GEV, to_remove, axis=0)
            assignment[assignment > to_remove] = (
                assignment[assignment > to_remove] - 1
            )

            fit = data[:, orphans].T @ cluster
            if ignore_polarity:
                fit = np.abs(fit)
            new_assignment = np.argmax(fit, axis=1)
            assignment[orphans] = new_assignment

            cluster_to_update = np.unique(new_assignment)
            for c in cluster_to_update:
                members = assignment == c
                if ignore_polarity:

                    old_weight = len(new_assignment == c) / members.sum()

                    sgn = np.sign(old_cluster @ cluster[:, c])

                    first_pc_init = (
                        old_weight * sgn * old_cluster
                        + (1.0 - old_weight) * cluster[:, c]
                    )

                    (
                        first_pc,
                        _,
                        converged,
                    ) = AAHCluster._first_principal_component(
                        data[:, members], first_pc_init, max_iter=n_chan
                    )
                    if converged:
                        logger.debug(
                            "AAHC %d/%d: " "power iterations converged.",
                            step,
                            n_steps,
                        )
                    else:
                        logger.debug(
                            "AAHC %d/%d "
                            "power iteration did not converge. "
                            "Computing fallback solution.",
                            step,
                            n_steps,
                        )
                        # fall back to covariance estimation
                        # and eigenvalue computation
                        data_cov = data[:, members] @ data[:, members].T
                        _, evecs = np.linalg.eigh(data_cov)
                        # eigenvector at largest eigenvalue
                        first_pc = evecs[:, -1]

                    cluster[:, c] = first_pc
                else:
                    cluster[:, c] = np.mean(data[:, members], axis=1)
                    cluster[:, c] /= np.linalg.norm(
                        cluster[:, c], axis=0, keepdims=True
                    )
                new_fit = cluster[:, slice(c, c + 1)] * data[:, members]
                if ignore_polarity:
                    new_fit = np.abs(new_fit)
                GEV[c] = np.sum(new_fit)
        return cluster.T, assignment

    # pylint: enable=too-many-locals

    @staticmethod
    @njit(cache=True)
    def _first_principal_component(
        X: NDArray[float],
        v0: NDArray[float],
        tol: float = 1e-6,
        max_iter: int = 100,
    ) -> Tuple[NDArray[float], float, bool]:
        """Compute first principal component.

        Parameters
        ----------
        X : numpy.ndarray
            Input data matrix with dimensions (channels x observations)
        v0 : numpy.ndarray
            Initial estimate of the principal vector with dimensions
            (channels,).
        tol : float (1e-6 by default)
            Tolerance for convergence.
        max_iter : int (100 by default)
            Maximum number of iterations to estimate the first principal
            component.

        Returns
        -------
        v : numpy.ndarray
            Estimated principal component vector.
        eig : float
            Eigenvalue of the covariance matrix of X (channels x channels)
        converged : bool
            False, if `max_iter` were reached.

        See :footcite:t:`Roweis1997` for additional information.
        """
        v = v0.flatten()
        assert v.shape[0] == X.shape[0]
        v /= np.linalg.norm(v)

        converged = False
        for _ in range(max_iter):
            s = np.sum((np.expand_dims(v, 0) @ X) * X, axis=1)

            eig = s.dot(v)
            s_norm = np.linalg.norm(s)

            if np.linalg.norm(eig * v - s) / s_norm < tol:
                converged = True
                break
            v = s / s_norm
        return v, eig, converged

    # --------------------------------------------------------------------

    @property
    def normalize_input(self) -> bool:
        """If set, the input data is normalized along the channel dimension.

        :type: `bool`
        """
        return self._normalize_input

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

    @staticmethod
    def _check_ignore_polarity(ignore_polarity: bool) -> bool:
        """Check that ignore_polarity is a boolean."""
        _check_type(ignore_polarity, (bool,), item_name="ignore_polarity")
        return ignore_polarity

    @staticmethod
    def _check_normalize_input(normalize_input: bool) -> bool:
        """Check that normalize_input is a boolean."""
        _check_type(normalize_input, (bool,), item_name="normalize_input")
        return normalize_input
