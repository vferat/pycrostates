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


def _check_lags(lags):
    _check_type(lags, (int, np.ndarray, list, tuple), "lags")
    if isinstance(lags, int):
        lags = np.arange(lags)
    elif isinstance(lags, (list, tuple)):
        lags = np.array(lags)
    if lags.ndim != 1:
        raise ValueError("Lags must be a 1D array.")
    if not np.issubdtype(lags.dtype, np.integer):
        raise ValueError("Lags values must be integers.")
    if not np.all(lags >= 0):
        raise ValueError("Lags values must be positive.")
    return lags


@fill_doc
def _joint_entropy(
    x: NDArray[int],
    y: NDArray[int],
    state_to_ignore: Optional[Union[int, None]] = -1,
    log_base: Optional[float, str] = 2,
)
    """
    Joint Shannon entropy of the symbolic sequences x, y.

    Parameters
    ----------
    x, y : array (n_symbols, )
        Symbolic sequences.
    %(state_to_ignore)s
    %(log_base)s

    Returns
    -------
    h: float
        joint Shannon entropy of (x, y).
    """
    _check_type(x, (np.ndarray,), "x")
    _check_type(y, (np.ndarray,), "y")
    if len(x) != len(y):
        raise ValueError("Sequences of different lengths.")
    _check_type(
        state_to_ignore,
        (
            int,
            None,
        ),
        "state_to_ignore",
    )
    log_base = _check_log_base(log_base)

    # ignoring state_to_ignore
    valid_indices = np.logical_and(x != state_to_ignore, y != state_to_ignore)
    x_valid = x[valid_indices]
    y_valid = y[valid_indices]

    # Compute the joint probability distribution
    n_clusters = np.max([x_valid, y_valid]) + 1
    joint_prob = np.zeros((n_clusters, n_clusters))
    for i in range(len(x_valid)):
        joint_prob[x_valid[i], y_valid[i]] += 1
    joint_prob /= len(x_valid)

    # Compute the joint entropy
    joint_entropy = scipy.stats.entropy(joint_prob.flatten(), base=log_base)
    return joint_entropy


### Auto information
@fill_doc
def _auto_information(
    labels: NDArray[int],
    k: int,
    state_to_ignore: Optional[Union[int, None]] = -1,
    log_base: Optional[Union[float, str]] = 2,
):
    """
    Computes the Auto-information for lag k.

    Parameters
    ----------
    %(labels_info)s
    %(k_info)s
    %(state_to_ignore)s
    %(log_base)s

    Returns
    -------
    a: float
        time-lagged auto-information for lag k.
    """
    _check_type(labels, (np.ndarray,), "labels")
    _check_type(k, (int,), "k")
    _check_type(state_to_ignore, (int,), "state_to_ignore")
    log_base = _check_log_base(log_base)

    n = len(labels)
    nmax = n - k
    h1 = entropy(
        labels[:nmax], state_to_ignore=state_to_ignore, log_base=log_base
    )
    h2 = entropy(
        labels[k : k + nmax],
        state_to_ignore=state_to_ignore,
        log_base=log_base,
    )
    h12 = _joint_entropy(
        labels[:nmax],
        labels[k : k + nmax],
        state_to_ignore=state_to_ignore,
        log_base=log_base,
    )
    a = h1 + h2 - h12
    return a


@fill_doc
def auto_information_function(
    segmentation: NDArray[int],
    lags: int,
    state_to_ignore: Optional[Union[int, None]] = -1,
    ignore_self: Optional[bool] = False,
    log_base: Optional[Union[float, str]] = 2,
    n_jobs: int = None,
):
    """
    Compute the Auto-information function (aif).

    TeX notation:
    I(X_{n+k} ; X_{n})
    = H(X_{n+k}) - H(X_{n+k} | X_{n})
    = H(X_{n+k}) - H(X_{n+k},X_{n}) + H(X_{n})
    = H(X_{n+k}) + H(X_{n}) - H(X_{n+k},X_{n})

    Parameters
    ----------
    %(labels_info)s
    %(k_info)s
    %(state_to_ignore)s
    %(log_base)s
    %(n_jobs)s

    Returns
    -------
    a: float (n_lags, )
        time-lagged mutual information array for each lag.
    lags: array (n_lags, )
        the lag values.
    """
    segmentation = _check_segmentation(segmentation)
    lags = _check_lags(lags)
    _check_type(state_to_ignore, (int,), "state_to_ignore")
    log_base = _check_log_base(log_base)
    n_jobs = _check_n_jobs(n_jobs)

    # ignore transition to itself (i.e. AAABBBBC -> ABC)
    if ignore_self:
        labels = np.array([s for s, _ in itertools.groupby(labels)])

    parallel, p_fun, _ = parallel_func(_auto_information, n_jobs, total=len(lags))
    runs = parallel(
        p_fun(segmentation, k, state_to_ignore=state_to_ignore, log_base=log_base)
        for k in lags
    )
    return runs, lags

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
    lags: Union[int,
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

    parallel, p_fun, _ = parallel_func(_partial_auto_information, n_jobs, total=len(lags))
    runs = parallel(
        p_fun(labels, k, state_to_ignore=state_to_ignore, log_base=log_base)
        for k in lags
    )
    return runs, lags
