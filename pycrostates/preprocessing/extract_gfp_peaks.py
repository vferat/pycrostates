"""Preprocessing functions to extract gfp peaks."""
from typing import Optional, Union

import numpy as np
from mne import BaseEpochs, pick_info
from mne.io import BaseRaw
from mne.io.pick import _picks_to_idx
from numpy.typing import NDArray
from scipy.signal import find_peaks

from .._typing import CHData, Picks
from ..utils._checks import (
    _check_picks_uniqueness,
    _check_reject_by_annotation,
    _check_tmin_tmax,
    _check_type,
)
from ..utils._docs import fill_doc
from ..utils._logs import logger, verbose


@fill_doc
@verbose
def extract_gfp_peaks(
    inst: Union[BaseRaw, BaseEpochs],
    picks: Picks = "eeg",
    return_all: bool = False,
    min_peak_distance: int = 2,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    reject_by_annotation: bool = True,
    verbose=None,
) -> CHData:
    """:term:`GFP` peaks extraction.

    Extract :term:`global field power` (GFP) peaks from :class:`~mne.Epochs` or
    :class:`~mne.io.Raw`.

    Parameters
    ----------
    inst : Raw | Epochs
        Instance from which to extract :term:`global field power` (GFP) peaks.
    picks : str | list | slice | None
        Channels to use for GFP computation.
        Note that all channels selected must have the
        same type. Slices and lists of integers will be interpreted as
        channel indices. In lists, channel name strings (e.g.
        ``['Fp1', 'Fp2']``) will pick the given channels. Can also be the
        string values “all” to pick all channels, or “data” to pick data
        channels. ``"eeg"`` (default) will pick all eeg channels.
        Note that channels in ``info['bads']`` will be included if their
        names or indices are explicitly provided.
    return_all : bool
        If True, output ChData instance will include all channels.
        If False, output ChData instance will only include channels
        used for GFP computation (i.e picks).
        Default to False.
    min_peak_distance : int
        Required minimal horizontal distance (``≥ 1`) in samples between
        neighboring peaks. Smaller peaks are removed first until the condition
        is fulfilled for all remaining peaks. Default to ``2``.
    %(tmin_raw)s
    %(tmax_raw)s
    %(reject_by_annotation_raw)s
    %(verbose)s

    Returns
    -------
    ch_data : ChData
        Samples at global field power peaks.
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
        reject_by_annotation = _check_reject_by_annotation(
            reject_by_annotation
        )

    # retrieve picks
    picks = _picks_to_idx(inst.info, picks, none="all", exclude="bads")
    picks_all = _picks_to_idx(inst.info, inst.ch_names, none="all", exclude="bads")
    _check_picks_uniqueness(inst.info, picks)
    # retrieve data array
    kwargs = (
        dict()
        if isinstance(inst, BaseEpochs)
        else dict(reject_by_annotation=reject_by_annotation)
    )
    data = inst.get_data(picks=picks, tmin=tmin, tmax=tmax, **kwargs)
    if return_all:
        data_all = inst.get_data(
            picks=picks_all, tmin=tmin, tmax=tmax, **kwargs
        )
    assert data.ndim in (2, 3)  # sanity-check

    # extract GFP peaks
    if data.ndim == 2:
        ind_peaks = _extract_gfp_peaks(data, min_peak_distance)
        if return_all:
            peaks = data_all[:, ind_peaks]
        else:
            peaks = data[:, ind_peaks]
    elif data.ndim == 3:
        peaks = list()  # run epoch per epoch
        for k in range(data.shape[0]):
            ind_peaks = _extract_gfp_peaks(data[k, :, :], min_peak_distance)
            print(ind_peaks)
            if return_all:
                peaks.append(data_all[k, :, ind_peaks].T) # TODO: why .T
            else:
                peaks.append(data[k, :, ind_peaks].T)
        peaks = np.hstack(peaks)

    n_samples = data.shape[-1]
    if isinstance(inst, BaseEpochs):
        n_samples *= len(inst)
    logger.info(
        "%s GFP peaks extracted out of %s samples (%.2f%% of the original "
        "data).",
        peaks.shape[1],
        n_samples,
        peaks.shape[1] / n_samples * 100,
    )

    if return_all:
        return ChData(peaks, pick_info(inst.info, picks_all))
    else:
        return ChData(peaks, pick_info(inst.info, picks))


def _extract_gfp_peaks(
    data: NDArray[float], min_peak_distance: int = 2
) -> NDArray[float]:
    """Extract GFP peaks from input data.

    Parameters
    ----------
    data : array of shape (n_channels, n_samples)
        The data to extract GFP peaks from.
    min_peak_distance : int
        Required minimal horizontal distance (>= 1) in samples between
        neighboring peaks. Smaller peaks are removed first until the condition
        is fulfilled for all remaining peaks. Default to 2.

    Returns
    -------
    peaks : array of shape (n_picks)
        The indices when peaks occur.
    """
    gfp = np.std(data, axis=0)
    peaks, _ = find_peaks(gfp, distance=min_peak_distance)
    return peaks
