"""Preprocessing functions to extract ChData from raw or epochs instances."""

from typing import Optional, Union

import numpy as np
from mne import BaseEpochs, pick_info
from mne.io import BaseRaw
from mne.io.pick import _picks_to_idx
from mne.preprocessing.ica import _check_start_stop
from numpy.typing import NDArray
from scipy.signal import find_peaks

from ..io import ChData
from ..utils._checks import (
    _check_reject_by_annotation,
    _check_tmin_tmax,
    _check_type,
)
from ..utils._docs import fill_doc
from ..utils._logs import logger, verbose


def _extract_gfp_peaks(
    data: NDArray[float], min_peak_distance: int = 2
) -> NDArray[float]:
    """Extract GFP peaks from input data.

    Parameters
    ----------
    data : array of shape (n_channels, n_samples)
        The data to extrat GFP peaks from.
    min_peak_distance : int
        Required minimal horizontal distance (>= 1) in samples between
        neighboring peaks. Smaller peaks are removed first until the condition
        is fulfilled for all remaining peaks. Default to 2.

    Returns
    -------
    data : array of shape (n_channels, n_samples)
        The data points at the GFP peaks.
    """
    gfp = np.std(data, axis=0)
    peaks, _ = find_peaks(gfp, distance=min_peak_distance)
    return data[:, peaks]


@fill_doc
@verbose
def extract_gfp_peaks(
    inst: Union[BaseRaw, BaseEpochs],
    picks: Optional[Union[str, NDArray[int]]] = None,
    min_peak_distance: int = 2,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    reject_by_annotation: bool = True,
    verbose=None,
) -> ChData:
    """GFP peaks extraction.

    Extract global field power peaks from :class:`mne.Epochs` or
    :class:`~mne.io.Raw`.

    Parameters
    ----------
    inst : Raw | Epochs
        Instance from which to extract GFP peaks.
    %(picks_all)s
    min_peak_distance : int
        Required minimal horizontal distance (>= 1) in samples between
        neighboring peaks. Smaller peaks are removed first until the condition
        is fulfilled for all remaining peaks. Default to 2.
    %(tmin_raw)s
    %(tmax_raw)s
    reject_by_annotation : bool
        Whether to reject by annotation. If True (default), segments annotated
        with description starting with ‘bad’ are omitted. If False, no
        rejection is done.
    %(verbose)s

    Returns
    -------
    ch_data : ChData
        Samples at global field power peaks.
    """
    _check_type(inst, (BaseRaw, BaseEpochs))
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

    # retrieve data array
    kwargs = (
        dict()
        if isinstance(inst, BaseEpochs)
        else dict(reject_by_annotation=reject_by_annotation)
    )
    data = inst.get_data(picks=picks, tmin=tmin, tmax=tmax, **kwargs)
    assert data.ndim in (2, 3)  # sanity-check

    # extract GFP peaks
    if data.ndim == 2:
        peaks = _extract_gfp_peaks(data, min_peak_distance)
    elif data.ndim == 3:
        peaks = list()  # run epoch per epoch
        for k in range(data.shape[0]):
            peaks.append(_extract_gfp_peaks(data[k, :, :], min_peak_distance))
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

    return ChData(peaks, pick_info(inst.info, picks))


@fill_doc
@verbose
def resample(
    inst,
    n_epochs=None,
    n_samples=None,
    coverage=None,
    replace=True,
    start=None,
    stop=None,
    reject_by_annotation=True,
    random_seed=None,
    verbose=None,
):
    """Resample recording into epochs of random samples.

    Resample :class:`~mne.io.Raw` or :class:`~mne.epochs.Epochs`
    into ``n_epochs`` each containing ``n_samples``
    random samples of the original recording.

    Parameters
    ----------
    inst : `mne.io.Raw`, `mne.Epochs` or `pycrostates.io.ChData`
        Instance from which to extract GFP peaks.
    n_epochs : int
        Number of epoch to draw.
    n_samples : int
        Length of each epoch (in samples).
    coverage: float (strictly positive)
        Ratio between resampling data size and size of the original recording.
        Can be > 1 if replace=True.
    replace: bool
        Whether or not to allow resampling with replacement.
        Default to True.
    reject_by_annotation : bool
        Whether to reject by annotation. If True (default), segments annotated
        with description starting with ‘bad’ are omitted. If False, no
        rejection is done.
    random_seed : float
        As resampling can be non-deterministic it can be useful to fix the
        random state to have reproducible results.
    %(tmin_raw)s
    %(tmax_raw)s
    %(verbose)s

    Returns
    -------
    ch_data : list of :class:`~pycrostates.io.ChData`
        List of resamples.

    Notes
    -----
    Only two of ``n_epochs``, ``n_samples`` and ``coverage``
    parameters must be defined, the non-defined one being
    computed during function execution.
    """
    from ..io import ChData

    _check_type(inst, (BaseRaw, BaseEpochs, ChData))

    if isinstance(inst, BaseRaw):
        reject_by_annotation = "omit" if reject_by_annotation else None
        start, stop = _check_start_stop(inst, start, stop)
        data = inst.get_data(
            start=start, stop=stop, reject_by_annotation=reject_by_annotation
        )

    if isinstance(inst, BaseEpochs):
        data = inst.get_data()
        data = np.hstack(data)

    if isinstance(inst, ChData):
        data = inst.data

    n_times = data.shape[1]

    if len([x for x in [n_epochs, n_samples, coverage] if x is None]) >= 2:
        raise ValueError(
            "At least two of the [n_epochs, n_samples, coverage] must be "
            "defined"
        )

    if coverage is not None:
        if coverage <= 0:
            raise ValueError("Coverage must be strictly positive")
    else:
        coverage = (n_epochs * n_samples) / n_times

    if n_epochs is None:
        n_epochs = int((n_times * coverage) / n_samples)

    if n_samples is None:
        n_samples = int((n_times * coverage) / n_epochs)

    if replace is False:
        if n_epochs * n_samples > n_times:
            raise ValueError(
                f"Can't draw {n_epochs} epochs of {n_samples} samples = "
                f"{n_epochs * n_samples} samples without replacement: "
                f"instance contains only {n_times} samples."
            )

    logger.info(
        "Resampling instance into %s epochs of %s covering %.2f%% of the "
        "original data.",
        n_epochs,
        n_samples,
        coverage * 100,
    )

    random_state = np.random.RandomState(random_seed)
    if replace:
        indices = random_state.randint(
            0, n_samples, size=(n_epochs, n_samples)
        )
    else:
        indices = np.arange(n_times)
        random_state.shuffle(indices)
        indices = indices[: n_epochs * n_samples]
        indices = indices.reshape((n_epochs, n_samples))

    data = data[:, indices]
    data = np.swapaxes(data, 0, 1)

    resamples = list()
    for d in data:
        resamples.append(ChData(d, inst.info))
    return resamples
