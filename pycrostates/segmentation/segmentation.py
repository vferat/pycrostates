"""Segmentation module for segmented data."""

from typing import Optional, Union

from matplotlib.axes import Axes
from mne import BaseEpochs
from mne.io import BaseRaw

from ..utils._checks import _check_type
from ..utils._docs import fill_doc
from ..viz import plot_epoch_segmentation, plot_raw_segmentation
from ._base import _BaseSegmentation


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
        show: Optional[bool] = None,
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
        %(show)s
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
            show=show,
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

    @fill_doc
    def plot(
        self,
        cmap: Optional[str] = None,
        axes: Optional[Axes] = None,
        cbar_axes: Optional[Axes] = None,
        *,
        block: bool = False,
        show: Optional[bool] = None,
        verbose=None,
    ):
        """Plot segmentation.

        Parameters
        ----------
        %(cmap)s
        %(axes_seg)s
        %(axes_cbar)s
        %(block)s
        %(show)s
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
            show=show,
            verbose=verbose,
        )

    # --------------------------------------------------------------------
    @property
    def epochs(self) -> BaseEpochs:
        """`~mne.Epochs` instance from which the segmentation was computed."""
        return self._inst.copy()
