"""Class and functions to use modified Kmeans."""

from __future__ import annotations  # c.f. PEP 563, PEP 649

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from mne import BaseEpochs
from mne.io import BaseRaw
from mne.parallel import parallel_func
from numpy.random import Generator, RandomState

from ..utils import _corr_vectors
from ..utils._checks import _check_n_jobs, _check_random_state, _check_type
from ..utils._docs import copy_doc, fill_doc
from ..utils._logs import logger
from ._base import _BaseCluster

if TYPE_CHECKING:
    from typing import Any, Optional, Union

    from .._typing import Picks, RandomState, ScalarFloatArray, ScalarIntArray
    from ..io import ChData


@fill_doc
class ModKMeans(_BaseCluster):
    r"""Modified K-Means clustering algorithm.

    See :footcite:t:`Marqui1995` for additional information.

    Parameters
    ----------
    %(n_clusters)s
    n_init : int
        Number of time the k-means algorithm is run with different centroid seeds. The
        final result will be the run with the highest Global Explained Variance (GEV).
    max_iter : int
        Maximum number of iterations of the K-means algorithm for a single run.
    tol : float
        Relative tolerance with regards estimate residual noise in the cluster centers
        of two consecutive iterations to declare convergence.
    %(random_state)s

    References
    ----------
    .. footbibliography::
    """

    # TODO: docstring for tol doesn't look english

    def __init__(
        self,
        n_clusters: int,
        n_init: int = 100,
        max_iter: int = 300,
        tol: Union[int, float] = 1e-6,
        random_state: RandomState = None,
    ):
        super().__init__()

        # k-means has a fix number of clusters defined at init
        self._n_clusters = _BaseCluster._check_n_clusters(n_clusters)
        self._cluster_names = [str(k) for k in range(self.n_clusters)]

        # k-means settings
        self._n_init = ModKMeans._check_n_init(n_init)
        self._max_iter = ModKMeans._check_max_iter(max_iter)
        self._tol = ModKMeans._check_tol(tol)
        self._random_state = _check_random_state(random_state)

        # fit variables
        self._GEV_ = None

    def _repr_html_(self, caption=None):
        from ..html_templates import repr_templates_env

        template = repr_templates_env.get_template("ModKMeans.html.jinja")
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
            n_init=self._n_init,
            GEV=GEV,
            cluster_names=self._cluster_names,
            fitted=self._fitted,
            n_samples=n_samples,
            ch_repr=ch_repr,
        )

    @copy_doc(_BaseCluster.__eq__)
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ModKMeans):
            if not super().__eq__(other):
                return False

            attributes = (
                "_n_init",
                "_max_iter",
                "_tol",
                # '_random_state',
                # TODO: think about comparison and I/O for random states
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

    @fill_doc
    def fit(
        self,
        inst: Union[BaseRaw, BaseEpochs, ChData],
        picks: Picks = "eeg",
        tmin: Optional[Union[int, float]] = None,
        tmax: Optional[Union[int, float]] = None,
        reject_by_annotation: bool = True,
        n_jobs: int = 1,
        *,
        verbose: Optional[str] = None,
    ) -> None:
        """Compute cluster centers.

        Parameters
        ----------
        inst : Raw | Epochs | ChData
            MNE `~mne.io.Raw`, `~mne.Epochs` or `~pycrostates.io.ChData` object from
            which to extract :term:`cluster centers`.
        picks : str | list | slice | None
            Channels to include. Note that all channels selected must have the same
            type. Slices and lists of integers will be interpreted as channel indices.
            In lists, channel name strings (e.g. ``['Fp1', 'Fp2']``) will pick the given
            channels. Can also be the string values ``“all”`` to pick all channels, or
            ``“data”`` to pick data channels. ``"eeg"`` (default) will pick all eeg
            channels. Note that channels in ``info['bads']`` will be included if their
            names or indices are explicitly provided.
        %(tmin_raw)s
        %(tmax_raw)s
        %(reject_by_annotation_raw)s
        %(n_jobs)s
        %(verbose)s
        """
        n_jobs = _check_n_jobs(n_jobs)
        data = super().fit(
            inst,
            picks=picks,
            tmin=tmin,
            tmax=tmax,
            reject_by_annotation=reject_by_annotation,
            verbose=verbose,
        )

        inits = self._random_state.randint(
            low=0, high=100 * self._n_init, size=(self._n_init)
        )

        if n_jobs == 1:
            best_gev, best_maps, best_segmentation = None, None, None
            count_converged = 0
            for init in inits:
                gev, maps, segmentation, converged = ModKMeans._kmeans(
                    data, self._n_clusters, self._max_iter, init, self._tol
                )
                if not converged:
                    continue
                if best_gev is None or gev > best_gev:
                    best_gev, best_maps, best_segmentation = (
                        gev,
                        maps,
                        segmentation,
                    )
                count_converged += 1
        else:
            parallel, p_fun, _ = parallel_func(
                ModKMeans._kmeans, n_jobs, total=self._n_init
            )
            runs = parallel(
                p_fun(data, self._n_clusters, self._max_iter, init, self._tol)
                for init in inits
            )
            try:
                best_run = np.nanargmax([run[0] if run[3] else np.nan for run in runs])
                best_gev, best_maps, best_segmentation, _ = runs[best_run]
                count_converged = sum(run[3] for run in runs)
            except ValueError:
                best_gev, best_maps, best_segmentation = None, None, None
                count_converged = 0

        if best_gev is not None:
            logger.info(
                "Selecting run with highest GEV = %.2f%% after %i/%i "
                "iterations converged.",
                best_gev * 100,
                count_converged,
                self._n_init,
            )
        else:
            logger.error(
                "All the K-means run failed to converge. Please adapt the tolerance "
                "and the maximum number of iteration."
            )
            self.fitted = False  # reset variables related to fit
            return  # break early

        self._GEV_ = best_gev
        self._cluster_centers_ = best_maps
        self._labels_ = best_segmentation
        self._fitted = True
        self._ignore_polarity = True

    @copy_doc(_BaseCluster.save)
    def save(self, fname: Union[str, Path]):
        super().save(fname)
        # TODO: to be replaced by a general writer than infers the writer from
        # the file extension.
        from ..io.fiff import _write_cluster  # pylint: disable=C0415

        _write_cluster(
            fname,
            self._cluster_centers_,
            self._info,
            "ModKMeans",
            self._cluster_names,
            self._fitted_data,
            self._labels_,
            n_init=self._n_init,
            max_iter=self._max_iter,
            tol=self._tol,
            GEV_=self._GEV_,
        )

    # --------------------------------------------------------------------
    @staticmethod
    def _kmeans(
        data: ScalarFloatArray,
        n_clusters: int,
        max_iter: int,
        random_state: Union[RandomState, Generator],
        tol: Union[int, float],
    ) -> tuple[float, ScalarFloatArray, ScalarIntArray, bool]:
        """Run the k-means algorithm."""
        gfp_sum_sq = np.sum(data**2)
        maps, converged = ModKMeans._compute_maps(
            data, n_clusters, max_iter, random_state, tol
        )
        activation = maps.dot(data)
        segmentation = np.argmax(np.abs(activation), axis=0)
        map_corr = _corr_vectors(data, maps[segmentation].T)
        gev = np.sum((data * map_corr) ** 2) / gfp_sum_sq
        return gev, maps, segmentation, converged

    @staticmethod
    def _compute_maps(
        data: ScalarFloatArray,
        n_clusters: int,
        max_iter: int,
        random_state: Union[RandomState, Generator],
        tol: Union[int, float],
    ) -> tuple[ScalarFloatArray, bool]:
        """Compute microstates maps.

        Based on mne_microstates by Marijn van Vliet <w.m.vanvliet@gmail.com>
        https://github.com/wmvanvliet/mne_microstates/blob/master/microstates.py
        """
        # TODO: Does this work if the RandomState is a generator?
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)

        # ------------------------- handle zeros maps -------------------------
        # zero map can be due to non data in the recording, it's unlikely that
        # all channels recorded the same value at the same time (=0 due to
        # average reference)
        # ---------------------------------------------------------------------
        data = data[:, np.linalg.norm(data.T, axis=1) != 0]
        n_channels, n_samples = data.shape
        data_sum_sq = np.sum(data**2)

        # Select random time points for our initial topographic maps
        init_times = random_state.choice(n_samples, size=n_clusters, replace=False)
        maps = data[:, init_times].T
        # Normalize the maps
        maps /= np.linalg.norm(maps, axis=1, keepdims=True)

        prev_residual = np.inf
        for _ in range(max_iter):
            # Assign each sample to the best matching microstate
            activation = maps.dot(data)
            segmentation = np.argmax(np.abs(activation), axis=0)

            # Recompute the topographic maps of the microstates, based on the
            # samples that were assigned to each state.
            for state in range(n_clusters):
                idx = segmentation == state
                if np.sum(idx) == 0:
                    maps[state] = 0
                    continue

                # Find largest eigenvector
                maps[state] = data[:, idx].dot(activation[state, idx])
                maps[state] /= np.linalg.norm(maps[state])

            # Estimate residual noise
            act_sum_sq = np.sum(np.sum(maps[segmentation].T * data, axis=0) ** 2)
            residual = abs(data_sum_sq - act_sum_sq)
            residual /= float(n_samples * (n_channels - 1))

            # check convergence
            if (prev_residual - residual) < (tol * residual):
                converged = True
                break

            prev_residual = residual

        else:
            converged = False

        return maps, converged

    # --------------------------------------------------------------------
    @property
    def n_init(self) -> int:  # noqa: D401
        """Number of k-means algorithms run with different centroid seeds.

        :type: `int`
        """
        return self._n_init

    @property
    def max_iter(self) -> int:
        """Maximum number of iterations of the k-means algorithm for a run.

        :type: `int`
        """
        return self._max_iter

    @property
    def tol(self) -> Union[int, float]:
        """Relative tolerance to reach convergence.

        :type: `float`
        """
        return self._tol

    @property
    def random_state(self) -> Union[RandomState, Generator]:
        """Random state to fix seed generation.

        :type: `~numpy.random.RandomState` | `~numpy.random.Generator`
        """
        return self._random_state

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

    # --------------------------------------------------------------------
    # For now, check function are defined as static within KMeans. If they are
    # used outside KMeans, they should be moved to regular function in _base.py
    # --------------------------------------------------------------------
    @staticmethod
    def _check_n_init(n_init: int) -> int:
        """Check that n_init is a positive integer."""
        _check_type(n_init, ("int",), item_name="n_init")
        if n_init <= 0:
            raise ValueError(
                "The number of initialization must be a positive integer. "
                f"Provided: '{n_init}'."
            )
        return n_init

    @staticmethod
    def _check_max_iter(max_iter: int) -> int:
        """Check that max_iter is a positive integer."""
        _check_type(max_iter, ("int",), item_name="max_iter")
        if max_iter <= 0:
            raise ValueError(
                "The number of max iteration must be a positive integer. "
                f"Provided: '{max_iter}'."
            )
        return max_iter

    @staticmethod
    def _check_tol(tol: Union[int, float]) -> Union[int, float]:
        """Check that tol is a positive number."""
        _check_type(tol, ("numeric",), item_name="tol")
        if tol <= 0:
            raise ValueError(
                f"The tolerance must be a positive number. Provided: '{tol}'."
            )
        return tol
