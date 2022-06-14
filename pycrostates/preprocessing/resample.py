"""Preprocessing functions to create resamples from raw or epochs instances."""

from typing import List, Optional, Union

import numpy as np
from mne import BaseEpochs, pick_info
from mne.io import BaseRaw
from mne.io.pick import _picks_to_idx

from .._typing import CHData, Picks, RANDomState
from ..utils._checks import (
    _check_random_state,
    _check_reject_by_annotation,
    _check_tmin_tmax,
    _check_type,
)
from ..utils._docs import fill_doc
from ..utils._logs import logger, verbose


@fill_doc
@verbose
def resample(
    inst: Union[BaseRaw, BaseEpochs, CHData],
    picks: Picks = None,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    reject_by_annotation: bool = True,
    n_resamples: int = None,
    n_samples: int = None,
    coverage: float = None,
    replace: bool = True,
    random_state: RANDomState = None,
    verbose=None,
) -> List[CHData]:
    """Resample a recording into epochs of random samples.

    Resample :class:`~mne.io.Raw`. :class:`~mne.Epochs` or
    `~pycrostates.io.ChData` into ``n_resamples`` each containing ``n_samples``
    random samples of the original recording.

    Parameters
    ----------
    inst : Raw | Epochs | ChData
        Instance to resample.
    %(picks_all)s
    %(tmin_raw)s
    %(tmax_raw)s
    %(reject_by_annotation_raw)s
    n_resamples : int
        Number of resamples to draw. Each epoch can be used to fit a separate
        clustering solution. See notes for additional information.
    n_samples : int
        Length of each epoch (in samples). See notes for additional
        information.
    coverage : float (strictly positive)
        Ratio between resampling data size and size of the original recording.
        See notes for additional information.
    replace : bool
        Whether or not to allow resampling with replacement.
    %(random_state)s
    %(verbose)s

    Returns
    -------
    resamples : list of :class:`~pycrostates.io.ChData`
        List of resamples.

    Notes
    -----
    Only two of ``n_resamples``, ``n_samples`` and ``coverage``
    parameters must be defined, the non-defined one will be
    determine at runtime by the 2 other parameters.
    """
    from ..io import ChData

    _check_type(inst, (BaseRaw, BaseEpochs, ChData))
    if isinstance(inst, (BaseRaw, BaseEpochs)):
        tmin, tmax = _check_tmin_tmax(inst, tmin, tmax)
    if isinstance(inst, BaseRaw):
        reject_by_annotation = _check_reject_by_annotation(
            reject_by_annotation
        )
    _check_type(n_resamples, (None, "int"), "n_resamples")
    _check_type(n_samples, (None, "int"), "n_samples")
    _check_type(coverage, (None, "numeric"), "coverage")
    _check_type(replace, (bool,), "replace")
    random_state = _check_random_state(random_state)

    # Check n_samples, coverage
    if n_resamples is not None:
        if n_resamples <= 0:
            raise ValueError(
                "Argument 'n_resamples' must be a strictly positive integer. "
                f"Provided: '{n_resamples}'."
            )
        if coverage is None and n_samples is None:
            raise ValueError(
                "When providing 'n_resamples', at least one of 'coverage' "
                "or 'n_samples' must be provided."
            )
        if coverage is not None and n_samples is not None:
            raise ValueError(
                "When providing 'n_resamples', only one of 'coverage' "
                "or 'n_samples' must be provided."
            )
        if coverage is not None and (coverage <= 0 or coverage > 1):
            raise ValueError(
                "Argument 'coverage' must respect 0 <= coverage <= 1. "
                f"Provided: '{coverage}'."
            )
        if n_samples is not None and n_samples <= 0:
            raise ValueError(
                "Argument 'n_samples' must be a strictly positive integer. "
                f"Provided: '{n_samples}'."
            )
    else:
        if coverage is None or n_samples is None:
            raise ValueError(
                "When 'n_resamples' is None, both 'coverage' and "
                "'n_samples' must be provided."
            )
        if n_samples <= 0:
            raise ValueError(
                "Argument 'n_samples' must be a strictly positive integer. "
                f"Provided: '{n_samples}'."
            )
        if coverage <= 0 or coverage > 1:
            raise ValueError(
                "Argument 'coverage' must respect 0 <= coverage <= 1. "
                f"Provided: '{coverage}'."
            )

    # retrieve picks
    picks = _picks_to_idx(inst.info, picks, none="all", exclude="bads")

    # retrieve data
    kwargs = dict() if isinstance(inst, ChData) else dict(tmin=tmin, tmax=tmax)
    if isinstance(inst, BaseRaw):
        kwargs["reject_by_annotation"] = reject_by_annotation
    data = inst.get_data(picks=picks, **kwargs)
    assert data.ndim in (2, 3)  # sanity-check
    if isinstance(inst, BaseEpochs):
        data = np.hstack(data)
    assert data.ndim == 2  # sanity-check
    n_times = data.shape[1]

    # Compute coverage / n_samples from the second
    if n_resamples is None:
        n_resamples = int((n_times * coverage) / n_samples)

    if n_samples is None:
        n_samples = int((n_times * coverage) / n_resamples)

    if coverage is None:
        coverage = (n_resamples * n_samples) / n_times

    if replace is False:
        if n_resamples * n_samples > n_times:
            raise ValueError(
                f"Can not draw {n_resamples} resamples of {n_samples} "
                f"samples = {n_resamples * n_samples} samples without "
                f"replacement because the instance contains only "
                f"{n_times} samples."
            )

    logger.info(
        "Resampling instance into %s resamples of %s samples "
        "covering %.2f%% of the original data.",
        n_resamples,
        n_samples,
        coverage * 100,
    )

    # random selection
    times_idx = np.arange(n_times)
    indices = random_state.choice(
        times_idx, size=n_resamples * n_samples, replace=replace
    )
    indices = indices.reshape((n_resamples, n_samples))

    # select data
    data = data[:, indices]
    data = np.swapaxes(data, 0, 1)

    # create list of ChData
    info = pick_info(inst.info, picks)
    resamples = list()
    for d in data:
        resamples.append(ChData(d, info))
    return resamples
