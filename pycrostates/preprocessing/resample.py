"""Preprocessing functions to extract ChData from raw or epochs instances."""

from typing import Optional, Union

import numpy as np
from mne import BaseEpochs, pick_info
from mne.io import BaseRaw
from mne.io.pick import _picks_to_idx
from numpy.random import Generator, RandomState
from numpy.typing import NDArray

from ..io import ChData
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
    inst: Union[BaseRaw, BaseEpochs, ChData],
    picks: Optional[Union[str, NDArray[int]]] = None,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    reject_by_annotation: bool = True,
    n_epochs: int = None,
    n_samples: int = None,
    coverage: float = None,
    replace: bool = True,
    random_state: Optional[Union[int, RandomState, Generator]] = None,
    verbose=None,
) -> ChData:
    """Resample recording into epochs of random samples.

    Resample :class:`~mne.io.Raw`. :class:`~mne.Epochs` or
    `~pycrostates.io.ChData` into ``n_epochs`` each containing ``n_samples``
    random samples of the original recording.

    Parameters
    ----------
    inst : Raw | Epochs | ChData
        Instance from which to extract GFP peaks.
    %(picks_all)s
    %(tmin_raw)s
    %(tmax_raw)s
    reject_by_annotation : bool
        Whether to reject by annotation. If True (default), segments annotated
        with description starting with ‘bad’ are omitted. If False, no
        rejection is done.
    n_epochs : int
        Number of epochs to draw.
    n_samples : int
        Length of each epoch (in samples).
    coverage : float (strictly positive)
        Ratio between resampling data size and size of the original recording.
        Can be > 1 if replace=True.
    replace : bool
        Whether or not to allow resampling with replacement.
    %(random_state)s
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
    _check_type(inst, (BaseRaw, BaseEpochs, ChData))
    if isinstance(inst, (BaseRaw, BaseEpochs)):
        tmin, tmax = _check_tmin_tmax(inst, tmin, tmax)
    if isinstance(inst, BaseRaw):
        reject_by_annotation = _check_reject_by_annotation(
            reject_by_annotation
        )
    _check_type(n_epochs, ("int", None), "n_epochs")
    _check_type(n_samples, ("int", None), "n_samples")
    _check_type(coverage, ("numeric", None), "coverage")
    _check_type(replace, (bool,), "replace")
    random_state = _check_random_state(random_state)

    # checks for n_epochs, n_samples and coverage
    if len([x for x in [n_epochs, n_samples, coverage] if x is None]) < 2:
        raise ValueError(
            "At least two arguments among ('n_epochs', 'n_samples', "
            "'coverage') must be defined."
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
    n_times = data.shape[1]

    # additional checks for n_epochs, n_samples and coverage
    #
    # /!\ Can not work, n_epochs can be None
    #
    if coverage is not None:
        if coverage <= 0:
            raise ValueError(
                "Argument 'coverage' must be strictly positive. "
                f"Provided: '{coverage}'."
            )
    else:
        coverage = (n_epochs * n_samples) / n_times

    # /!\ Can not work, n_samples can be None
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
