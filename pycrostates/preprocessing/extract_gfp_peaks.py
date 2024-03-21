"""Preprocessing functions to extract gfp peaks."""

from __future__ import annotations  # c.f. PEP 563, PEP 649

from typing import TYPE_CHECKING

import numpy as np
from mne import BaseEpochs, pick_info
from mne.io import BaseRaw
from mne.utils import check_version
from scipy.signal import find_peaks

if check_version("mne", "1.6"):
    from mne._fiff.pick import _picks_to_idx
else:
    from mne.io.pick import _picks_to_idx

from ..utils._checks import (
    _check_picks_uniqueness,
    _check_reject_by_annotation,
    _check_tmin_tmax,
    _check_type,
)
from ..utils._docs import fill_doc
from ..utils._logs import logger, verbose

if TYPE_CHECKING:
    from typing import Optional, Union

    from .._typing import Picks, ScalarFloatArray
    from ..io import ChData


@fill_doc
@verbose
def extract_gfp_peaks(
    inst: Union[BaseRaw, BaseEpochs],
    picks: Picks = "eeg",
    return_all: bool = False,
    min_peak_distance: int = 1,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    reject_by_annotation: bool = True,
    verbose=None,
) -> ChData:
    """:term:`Global Field Power` (:term:`GFP`) peaks extraction.

    Extract :term:`Global Field Power` (:term:`GFP`) peaks from :class:`~mne.Epochs` or
    :class:`~mne.io.Raw`.

    Parameters
    ----------
    inst : Raw | Epochs
        Instance from which to extract :term:`global field power` (GFP) peaks.
    picks : str | list | slice | None
        Channels to use for GFP computation. Note that all channels selected must have
        the same type. Slices and lists of integers will be interpreted as channel
        indices. In lists, channel name strings (e.g. ``['Fp1', 'Fp2']``) will pick the
        given channels. Can also be the string values ``“all”`` to pick all channels, or
        ``“data”`` to pick data channels. ``"eeg"`` (default) will pick all eeg
        channels. Note that channels in ``info['bads']`` will be included if their
        names or indices are explicitly provided.
    return_all : bool
        If True, the returned `~pycrostates.io.ChData` instance will include all
        channels. If False (default), the returned `~pycrostates.io.ChData` instance
        will only include channels used for GFP computation (i.e ``picks``).
    min_peak_distance : int
        Required minimal horizontal distance (``≥ 1`) in samples between neighboring
        peaks. Smaller peaks are removed first until the condition is fulfilled for all
        remaining peaks. Default to ``1``.
    %(tmin_raw)s
    %(tmax_raw)s
    %(reject_by_annotation_raw)s
    %(verbose)s

    Returns
    -------
    ch_data : ChData
        Samples at global field power peaks.

    Notes
    -----
    The :term:`Global Field Power` (:term:`GFP`) peaks are extracted with
    :func:`scipy.signal.find_peaks`. Only the ``distance`` argument is filled with the
    value provided in ``min_peak_distance``. The other arguments are set to their
    default values.
    """
    from ..io import ChData

    _check_type(inst, (BaseRaw, BaseEpochs), "inst")
    _check_type(min_peak_distance, ("int",), "min_peak_distance")
    if min_peak_distance < 1:
        raise ValueError(
            "Argument 'min_peak_distance' must be superior or "
            f"equal to 1. Provided: {min_peak_distance}."
        )
    tmin, tmax = _check_tmin_tmax(inst, tmin, tmax)
    if isinstance(inst, BaseRaw):
        reject_by_annotation = _check_reject_by_annotation(reject_by_annotation)

    # retrieve picks
    picks = _picks_to_idx(inst.info, picks, none="all", exclude="bads")
    picks_all = _picks_to_idx(inst.info, inst.ch_names, none="all", exclude="bads")
    _check_picks_uniqueness(inst.info, picks)

    # set kwargs for .get_data()
    kwargs = dict(tmin=tmin, tmax=tmax)
    if isinstance(inst, BaseRaw):
        kwargs["reject_by_annotation"] = reject_by_annotation

    # extract GFP peaks
    if isinstance(inst, BaseRaw):
        # retrieve data array on which we look for GFP peaks
        data = inst.get_data(picks=picks, **kwargs)
        # retrieve indices of GFP peaks
        ind_peaks = _extract_gfp_peaks(data, min_peak_distance)
        # retrieve the peaks data
        if return_all:
            del data  # free up memory
            data = inst.get_data(picks=picks_all, **kwargs)
        peaks = data[:, ind_peaks]
    elif isinstance(inst, BaseEpochs):
        peaks = list()  # run epoch per epoch
        for k in range(len(inst)):  # pylint: disable=consider-using-enumerate
            data = inst[k].get_data(picks=picks, **kwargs)[0, :, :]
            # data is 2D, of shape (n_channels, n_samples)
            ind_peaks = _extract_gfp_peaks(data, min_peak_distance)
            if return_all:
                del data  # free up memory
                data = inst[k].get_data(picks=picks_all, **kwargs)[0, :, :]
            peaks.append(data[:, ind_peaks])
        peaks = np.hstack(peaks)

    n_samples = inst.times.size
    if isinstance(inst, BaseEpochs):
        n_samples *= len(inst)
    logger.info(
        "%s GFP peaks extracted out of %s samples (%.2f%% of the original data).",
        peaks.shape[1],
        n_samples,
        peaks.shape[1] / n_samples * 100,
    )

    info = pick_info(inst.info, picks_all if return_all else picks)
    return ChData(peaks, info)


def _extract_gfp_peaks(
    data: ScalarFloatArray, min_peak_distance: int = 2
) -> ScalarFloatArray:
    """Extract GFP peaks from input data.

    Parameters
    ----------
    data : array of shape (n_channels, n_samples)
        The data to extract GFP peaks from.
    min_peak_distance : int
        Required minimal horizontal distance (>= 1) in samples between neighboring
        peaks. Smaller peaks are removed first until the condition is fulfilled for all
        remaining peaks. Default to 2.

    Returns
    -------
    peaks : array of shape (n_picks,)
        The indices when peaks occur.
    """
    gfp = np.std(data, axis=0)
    peaks, _ = find_peaks(gfp, distance=min_peak_distance)
    return peaks
