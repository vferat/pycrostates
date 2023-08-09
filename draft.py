"""Partial autoinformation module for segmented data."""

# code from https://github.com/Frederic-vW/AIF-PAIF:
# F. von Wegner, Partial Autoinformation to Characterize Symbolic Sequences
# Front Physiol (2018) https://doi.org/10.3389/fphys.2018.01382
import itertools
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.stats
from mne.parallel import parallel_func
from numpy.typing import NDArray

from .._typing import Segmentation
from ..utils._checks import _check_n_jobs, _check_type, _check_value
from ..utils._docs import fill_doc

# TODO: should we normalize aif and paif?
# The n_clusters parameter should reflect the total number of possible states,
# including the ignored state, if applicable. If the ignored state is not
# considered as a valid state, it should be excluded from the count of
# n_clusters.


###  Partial auto-information
@fill_doc
def _partial_auto_information(
    labels: NDArray[int],
    k: int,
    state_to_ignore: Optional[Union[int, None]] = -1,
    log_base: Optional[Union[float, str]] = 2,
):
    """
    Compute the partial auto-information for lag k.

    Parameters
    ----------
    %(labels_info)s
    k: int
        the lag in sample.
    %(state_to_ignore)s
    %(log_base)s

    Returns
    -------
    p: float
        Partial auto-information for lag k.
    """
    if k <= 1:
        return _auto_information(
            labels, k, state_to_ignore=state_to_ignore, log_base=log_base
        )
    h1 = _joint_entropy_history(
        labels,
        k,
        state_to_ignore=state_to_ignore,
        log_base=log_base,
    )
    h2 = _joint_entropy_history(
        labels,
        k - 1,
        state_to_ignore=state_to_ignore,
        log_base=log_base,
    )
    h3 = _joint_entropy_history(
        labels,
        k + 1,
        state_to_ignore=state_to_ignore,
        log_base=log_base,
    )
    p = 2 * h1 - h2 - h3
    return p


@fill_doc
def partial_auto_information_function(
    segmentation: [Segmentation, NDArray[int]],
    lags: Union[
        int,
        List[int],
        Tuple[int, ...],
        NDArray[int],
    ],
    state_to_ignore: Optional[Union[int, None]] = -1,
    ignore_self: Optional[bool] = False,
    log_base: Optional[Union[float, str]] = 2,
    n_jobs: Optional[int] = 1,
):
    """
    Compute the Partial auto-information function.

    TeX notation:
    I(X_{n+k} ; X_{n} | X_{n+k-1}^{(k-1)})
    = H(X_{n+k} | X_{n+k-1}^{(k-1)}) - H(X_{n+k} | X_{n+k-1}^{(k-1)}, X_{n})
    =   H(X_{n+k},X_{n+k-1}^{(k-1)}) - H(X_{n+k-1}^{(k-1)})
      - H(X_{n+k},X_{n+k-1}^{(k)}) + H(X_{n+k-1}^{(k)})
    = H(X_{n+k}^{(k)}) - H(X_{n+k-1}^{(k-1)}) - H(X_{n+k}^{(k+1)}) + H(X_{n+k-1}^{(k)})

    Parameters
    ----------
    %(segmentation_or_labels)s
    %(lags)s
    %(log_base)s
    %(n_jobs)s

    Returns
    -------
    p: array (n_symbols, )
        Partial auto-information array for each lag.
    lags: array (n_lags, )
        the lag values.
    """
    labels = _check_segmentation(segmentation)
    lags = _check_lags(lags)
    _check_type(state_to_ignore, (int,), "state_to_ignore")
    _check_type(ignore_self, (bool,), "ignore_self")
    log_base = _check_log_base(log_base)
    n_jobs = _check_n_jobs(n_jobs)

    # ignore transition to itself (i.e. 1 -> 1)
    if ignore_self:
        labels = np.array([s for s, _ in itertools.groupby(labels)])

    parallel, p_fun, _ = parallel_func(
        _partial_auto_information, n_jobs, total=len(lags)
    )
    runs = parallel(
        p_fun(labels, k, state_to_ignore=state_to_ignore, log_base=log_base)
        for k in lags
    )
    return runs, lags
