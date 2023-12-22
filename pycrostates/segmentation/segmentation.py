"""Segmentation module for segmented data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib.axes import Axes
from mne import BaseEpochs
from mne.io import BaseRaw

from ..utils._checks import _check_type
from ..utils._docs import fill_doc
from ..viz import plot_epoch_segmentation, plot_raw_segmentation
from ._base import _BaseSegmentation

if TYPE_CHECKING:
    from typing import Optional, Union

    from pandas import DataFrame


@fill_doc
class RawSegmentation(_BaseSegmentation):
    """
    Contains the segmentation of a `~mne.io.Raw` instance.

    Parameters
    ----------
    %(labels_raw)s
    raw : Raw
        `~mne.io.Raw` instance used for prediction.
    %(cluster_centers_seg)s
    %(cluster_names)s
    %(predict_parameters)s
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _check_type(self._inst, (BaseRaw,), item_name="raw")
        if self._labels.ndim != 1:
            raise ValueError(
                "Argument 'labels' should be a 1D array. The provided array shape "
                f"is {self._labels.shape} which has {self._labels.ndim} dimensions."
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
        """`~mne.io.Raw` instance from which the segmentation was computed."""
        return self._inst.copy()


@fill_doc
class EpochsSegmentation(_BaseSegmentation):
    """Contains the segmentation of an `~mne.Epochs` instance.

    Parameters
    ----------
    %(labels_epo)s
    epochs : Epochs
        `~mne.Epochs` instance used for prediction.
    %(cluster_centers_seg)s
    %(cluster_names)s
    %(predict_parameters)s
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _check_type(self._inst, (BaseEpochs,), "epochs")

        if self._labels.ndim != 2:
            raise ValueError(
                "Argument 'labels' should be a 2D array. The provided array shape "
                f"is {self._labels.shape} which has {self._labels.ndim} dimensions."
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
                f"samples, while the 'labels' has {self._labels.shape[-1]} samples."
            )

    def __getitem__(self, item):
        """Select epochs in a :class:`~pycrostates.segmentation.EpochsSegmentation`.

        Parameters
        ----------
        item : slice, array-like, str, or list
            See below for use cases.

        Returns
        -------
        epochs : instance of EpochsSegmentation
            Returns a copy of the original instance. See below for use cases.

        Notes
        -----
        :class:`~pycrostates.segmentation.EpochsSegmentation` can be accessed as
        ``segmentation[...]`` in several ways:

        1. **Integer or slice:** ``segmentation[idx]`` will return an
           :class:`~pycrostates.segmentation.EpochsSegmentation` object with a subset of
           epochs chosen by index (supports single index and Python-style slicing).

        2. **String:** ``segmentation['name']`` will return an
           :class:`~pycrostates.segmentation.EpochsSegmentation` object comprising only
           the epochs labeled ``'name'`` (i.e., epochs created around events with the
           label ``'name'``).

           If there are no epochs labeled ``'name'`` but there are epochs
           labeled with /-separated tags (e.g. ``'name/left'``,
           ``'name/right'``), then ``segmentation['name']`` will select the epochs
           with labels that contain that tag (e.g., ``segmentation['left']`` selects
           epochs labeled ``'audio/left'`` and ``'visual/left'``, but not
           ``'audio_left'``).

           If multiple tags are provided *as a single string* (e.g.,
           ``segmentation['name_1/name_2']``), this selects epochs containing *all*
           provided tags. For example, ``segmentation['audio/left']`` selects
           ``'audio/left'`` and ``'audio/quiet/left'``, but not
           ``'audio/right'``. Note that tag-based selection is insensitive to
           order: tags like ``'audio/left'`` and ``'left/audio'`` will be
           treated the same way when selecting via tag.

        3. **List of strings:** ``segmentation[['name_1', 'name_2', ... ]]`` will
           return an :class:`~pycrostates.segmentation.EpochsSegmentation` object
           comprising epochs that match *any* of the provided names (i.e., the list of
           names is treated as an inclusive-or condition). If *none* of the provided
           names match any epoch labels, a ``KeyError`` will be raised.

           If epoch labels are /-separated tags, then providing multiple tags
           *as separate list entries* will likewise act as an inclusive-or
           filter. For example, ``segmentation[['audio', 'left']]`` would select
           ``'audio/left'``, ``'audio/right'``, and ``'visual/left'``, but not
           ``'visual/right'``.

        4. **Pandas query:** ``segmentation['pandas query']`` will return an
           :class:`~pycrostates.segmentation.EpochsSegmentation` object with a subset of
           epochs (and matching metadata) selected by the query called with
           ``self.metadata.eval``, e.g.::

               epochs["col_a > 2 and col_b == 'foo'"]

           would return all epochs whose associated ``col_a`` metadata was
           greater than two, and whose ``col_b`` metadata was the string 'foo'.
           Query-based indexing only works if Pandas is installed and
           ``self.metadata`` is a :class:`pandas.DataFrame`.
        """
        inst = self.copy()  # noqa: F841

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
        # error checking on the input is performed in the viz function.
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
        """`~mne.Epochs` instance from which the segmentation was computed."""
        return self._inst.copy()

    @property
    def metadata(self) -> Optional[DataFrame]:
        """Epochs metadata."""
        return self._inst.metadata
