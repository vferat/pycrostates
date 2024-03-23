"""Partial autoinformation module for segmented data.

Code from https://github.com/Frederic-vW/AIF-PAIF
F. von Wegner, Partial Autoinformation to Characterize Symbolic Sequences
Front Physiol (2018) https://doi.org/10.3389/fphys.2018.01382
"""

from __future__ import annotations  # c.f. PEP 563, PEP 649

import itertools
from typing import TYPE_CHECKING

import numpy as np
import scipy.stats
from mne.parallel import parallel_func

from ..utils._checks import _check_n_jobs, _check_type, _check_value, _ensure_int
from ..utils._docs import fill_doc

if TYPE_CHECKING:
    from typing import Optional, Union

    from .._typing import ScalarFloatArray, ScalarIntArray
    from ._base import _BaseSegmentation


def _check_log_base(log_base) -> float:
    _check_type(log_base, ("numeric", str), "log_base")
    if isinstance(log_base, str):
        mapping = {"bits": 2, "natural": np.e, "dits": 10}
        _check_value(
            log_base,
            mapping,
            "log_base",
            extra="when string is provided",
        )
        log_base = mapping[log_base]
    if log_base <= 0:
        raise ValueError(
            f"If numeric, 'log_base' must be a positive number. '{log_base}' is "
            "invalid."
        )
    return log_base


def _check_lags(lags) -> ScalarIntArray:
    _check_type(lags, ("int", "array-like"), "lags")
    if isinstance(lags, int):
        if lags < 1:
            raise ValueError("If integer, lags must be >= 1.")
        lags = np.arange(lags)
    elif not isinstance(lags, np.ndarray):
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
    x: ScalarIntArray,
    y: ScalarIntArray,
    state_to_ignore: Optional[int] = -1,
    log_base: Union[float, str] = 2,
) -> float:
    """Joint Shannon entropy of the symbolic sequences x, y.

    Parameters
    ----------
    x : array (n_symbols, )
        Symbolic sequence.
    y : array (n_symbols, )
        Symbolic sequence.
    %(state_to_ignore)s
    %(log_base)s

    Returns
    -------
    h : float
        joint Shannon entropy of (x, y).
    """
    _check_type(x, (np.ndarray,), "x")
    _check_type(y, (np.ndarray,), "y")
    if any(elt.ndim != 1 for elt in (x, y)):
        raise ValueError("Sequences must be 1D arrays.")
    if x.size != y.size:
        raise ValueError("Sequences of different lengths.")
    if state_to_ignore is not None:
        state_to_ignore = _ensure_int(state_to_ignore, "state_to_ignore")
    log_base = _check_log_base(log_base)

    # ignoring state_to_ignore
    valid_indices = np.logical_and(x != state_to_ignore, y != state_to_ignore)
    x_valid = x[valid_indices]
    y_valid = y[valid_indices]

    # compute the joint probability distribution
    n_clusters = np.max([x_valid, y_valid]) + 1
    joint_prob = np.zeros((n_clusters, n_clusters))
    for i in range(len(x_valid)):
        joint_prob[x_valid[i], y_valid[i]] += 1
    joint_prob /= len(x_valid)

    return scipy.stats.entropy(joint_prob.flatten(), base=log_base)


def _check_labels(labels, item_name: str = "labels") -> None:
    _check_type(labels, (np.ndarray,), item_name)
    if labels.ndim != 1:
        raise ValueError(f"{item_name} must be a 1D array.")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"{item_name} must be an array of integers.")


def _check_segmentation(
    segmentation, item_name: str = "segmentation"
) -> ScalarIntArray:
    from ._base import _BaseSegmentation

    _check_type(segmentation, (_BaseSegmentation,), item_name)
    return segmentation._labels.reshape(-1)  # reshape if epochs (returns a view)


@fill_doc
def _joint_entropy_history(
    labels: ScalarIntArray,
    k: int,
    state_to_ignore: Optional[int] = -1,
    log_base: Union[float, str] = 2,
) -> float:
    r"""Compute the joint Shannon of k-histories x[t:t+k].

    Compute the joint Shannon entropy of the k-histories x[t:t+k].

    Parameters
    ----------
    %(labels_info)s
    %(state_to_ignore)s
    %(log_base)s

    Returns
    -------
    h : float
        The Shannon entropy of the sequence labels[t:t+k].
    """
    _check_labels(labels)
    _check_type(k, ("int",), "k")
    if state_to_ignore is not None:
        state_to_ignore = _ensure_int(state_to_ignore, "state_to_ignore")
    log_base = _check_log_base(log_base)

    # Construct the k-history sequences while ignoring the state
    histories = []
    for i in range(len(labels) - k + 1):
        history = tuple(labels[i : i + k])
        if state_to_ignore not in history:
            histories.append(history)

    n_clusters = np.max(labels) + 1
    # Compute the joint probability distribution
    joint_dist = np.zeros(tuple(k * [n_clusters]))
    for i in range(len(labels) - k + 1):  # TODO: check +1 (not in original code)
        history = tuple(labels[i : i + k])
        if state_to_ignore not in history:
            joint_dist[history] += 1.0

    return scipy.stats.entropy(joint_dist.flatten(), base=log_base)


@fill_doc
def _entropy(
    labels: ScalarIntArray,
    state_to_ignore: Optional[int] = -1,
    log_base: Union[float, str] = 2,
) -> float:
    r"""Compute the Shannon entropy of the a symbolic sequence.

    Parameters
    ----------
    %(labels_info)s
    %(state_to_ignore)s
    %(ignore_repetitions)s
    %(log_base)s

    Returns
    -------
    h : float
        The Shannon entropy of the sequence.
    """
    _check_labels(labels)
    if state_to_ignore is not None:
        _ensure_int(state_to_ignore, "state_to_ignore")
    log_base = _check_log_base(log_base)
    return _joint_entropy_history(
        labels, k=1, state_to_ignore=state_to_ignore, log_base=log_base
    )


@fill_doc
def entropy(
    segmentation: _BaseSegmentation,
    ignore_repetitions: bool = False,
    log_base: Union[float, str] = 2,
) -> float:
    r"""Compute the Shannon entropy of a symbolic sequence.

    Compute the Shannon entropy\ :footcite:p:`shannon1948mathematical` of the microstate
    symbolic sequence.

    Parameters
    ----------
    %(segmentation)s
    %(ignore_repetitions)s
    %(log_base)s

    Returns
    -------
    h : float
        The Shannon entropy of the sequence.

    References
    ----------
    .. footbibliography::
    """
    labels = _check_segmentation(segmentation)
    _check_type(ignore_repetitions, (bool,), "ignore_repetitions")
    log_base = _check_log_base(log_base)
    if ignore_repetitions:  # ignore transition to itself (i.e. AAABBBBC -> ABC)
        labels = np.array([s for s, _ in itertools.groupby(labels)])
    return _entropy(labels, state_to_ignore=-1, log_base=log_base)


# -- excess entropy rate ---------------------------------------------------------------
@fill_doc
def _excess_entropy_rate(
    labels: ScalarIntArray,
    history_length: int,
    state_to_ignore: Optional[int] = -1,
    log_base: Union[float, str] = 2,
    n_jobs: int = 1,
) -> tuple[float, float, float, ScalarIntArray, ScalarFloatArray]:
    """Estimate the entropy rate and the excess_entropy from a linear fit.

    Parameters
    ----------
    %(labels_info)s
    history_length : int
        Maximum history length in sample to estimate the excess entropy rate.
    %(state_to_ignore)s
    %(log_base)s
    %(n_jobs)s

    Returns
    -------
    a : float
        Entropy rate (slope).
    b : float
        Excess entropy (intercept).
    residual: float
        Sum of squared residuals of the least squares fit.
    lags : array of shape (history_length,)
        Lag values in sample used for the fit.
    joint_entropies : array of shape (history_length,)
        Joint entropy value for each lag.
    """
    _check_labels(labels)
    if state_to_ignore is not None:
        state_to_ignore = _ensure_int(state_to_ignore, "state_to_ignore")
    lags = np.arange(1, history_length + 1)
    parallel, p_fun, _ = parallel_func(_joint_entropy_history, n_jobs, total=len(lags))
    runs = parallel(
        p_fun(labels, k, state_to_ignore=state_to_ignore, log_base=log_base)
        for k in lags
    )

    (a, b), residuals, _, _, _ = np.polyfit(lags, runs, 1, full=True)
    return (a, b, residuals[0], lags, runs)


@fill_doc
def excess_entropy_rate(
    segmentation: _BaseSegmentation,
    history_length: int,
    ignore_repetitions: bool = False,
    log_base: Union[float, str] = 2,
    n_jobs: int = 1,
) -> tuple[float, float, float, ScalarIntArray, ScalarFloatArray]:
    r"""Estimate the entropy rate and the ``excess_entropy`` of the segmentation.

    The entropy rate and the ``excess_entropy`` are estimated from a linear fit:

    .. math::

        H(X_{n}^{(k)}) = a \cdot k + b

    where ``a`` is the entropy rate and ``b`` the excess entropy
    as described in :footcite:t:`von2018partial`.

    Parameters
    ----------
    %(segmentation)s
    history_length : int
        Maximum history length in sample to estimate the excess entropy rate.
    %(ignore_repetitions)s
    %(log_base)s
    %(n_jobs)s

    Returns
    -------
    a : float
        Entropy rate (slope).
    b : float
        Excess entropy (intercept).
    residual : float
        Sum of squared residuals of the least squares fit.
    lags : array of shape (history_length,)
        Lag values in sample used for the fit.
    joint_entropies : array of shape (history_length,)
        Joint entropy value for each lag.

    References
    ----------
    .. footbibliography::
    """
    labels = _check_segmentation(segmentation)
    _check_type(history_length, (int,), "history_length")
    _check_type(ignore_repetitions, (bool,), "ignore_repetitions")
    log_base = _check_log_base(log_base)
    n_jobs = _check_n_jobs(n_jobs)
    if ignore_repetitions:  # ignore transition to itself (i.e. 1 -> 1)
        labels = np.array([s for s, _ in itertools.groupby(labels)])
    return _excess_entropy_rate(
        labels,
        history_length,
        state_to_ignore=-1,
        log_base=log_base,
        n_jobs=n_jobs,
    )


@fill_doc
def _auto_information(
    labels: ScalarIntArray,
    k: int,
    state_to_ignore: Optional[int] = -1,
    log_base: Union[float, str] = 2,
) -> float:
    """Compute the Auto-information for lag k.

    Parameters
    ----------
    %(labels_info)s
    k : int
        Lag value in sample.
    %(state_to_ignore)s
    %(log_base)s

    Returns
    -------
    a : float
        time-lagged auto-information for lag k.
    """
    _check_labels(labels)
    k = _ensure_int(k, "k")
    if state_to_ignore is not None:
        state_to_ignore = _ensure_int(state_to_ignore, "state_to_ignore")
    log_base = _check_log_base(log_base)

    nmax = len(labels) - k
    h1 = _entropy(labels[:nmax], state_to_ignore=state_to_ignore, log_base=log_base)
    h2 = _entropy(
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
    segmentation: _BaseSegmentation,
    lags: Union[
        int,
        list[int],
        tuple[int, ...],
        ScalarIntArray,
    ],
    ignore_repetitions: bool = False,
    log_base: Union[float, str] = 2,
    n_jobs: int = 1,
) -> tuple[ScalarIntArray, ScalarFloatArray]:
    r"""Compute the Auto-information function (aif).

    Compute the Auto-information function (aif) as described
    in :footcite:t:`von2018partial`:

    .. math::

        I(X_{n+k} ; X_{n})
        = H(X_{n+k}) - H(X_{n+k} | X_{n})
        = H(X_{n+k}) - H(X_{n+k},X_{n}) + H(X_{n})
        = H(X_{n+k}) + H(X_{n}) - H(X_{n+k},X_{n})

    Parameters
    ----------
    %(segmentation)s
    lags : int | list | tuple | array of shape ``(n_lags,)``
        Lags at which to compute the auto-information function.
        If int, will use ``lags = np.arange(lags)``.
    %(ignore_repetitions)s
    %(log_base)s
    %(n_jobs)s

    Returns
    -------
    lags : array of shape ``(n_lags,)``
        Lag values in sample.
    ai : array of shape ``(n_lags,)``
        Time-lagged mutual information array for each lag.

    References
    ----------
    .. footbibliography::
    """  # noqa: E501
    labels = _check_segmentation(segmentation)
    lags = _check_lags(lags)
    log_base = _check_log_base(log_base)
    n_jobs = _check_n_jobs(n_jobs)
    if ignore_repetitions:  # ignore transition to itself (i.e. AAABBBBC -> ABC)
        labels = np.array([s for s, _ in itertools.groupby(labels)])
    parallel, p_fun, _ = parallel_func(_auto_information, n_jobs, total=len(lags))
    runs = parallel(
        p_fun(labels, k, state_to_ignore=-1, log_base=log_base) for k in lags
    )
    ai = np.array(runs)
    return lags, ai


@fill_doc
def _partial_auto_information(
    labels: ScalarIntArray,
    k: int,
    state_to_ignore: Optional[int] = -1,
    log_base: Union[float, str] = 2,
) -> float:
    """Compute the partial auto-information for lag k.

    Parameters
    ----------
    %(labels_info)s
    k : int
        Lag values in sample.
    %(state_to_ignore)s
    %(log_base)s

    Returns
    -------
    p : float
        Partial auto-information for lag k.
    """
    _check_labels(labels)
    if state_to_ignore is not None:
        state_to_ignore = _ensure_int(state_to_ignore, "state_to_ignore")
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
    segmentation: _BaseSegmentation,
    lags: Union[
        int,
        list[int],
        tuple[int, ...],
        ScalarIntArray,
    ],
    ignore_repetitions: bool = False,
    log_base: Union[float, str] = 2,
    n_jobs: Optional[int] = 1,
) -> tuple[ScalarIntArray, ScalarFloatArray]:
    r"""Compute the Partial auto-information function.

    Compute the Partial auto-information function as described
    in :footcite:t:`von2018partial`.

    .. math::

        I(X_{n+k} ; X_{n} | X_{n+k-1}^{(k-1)})
        = H(X_{n+k} | X_{n+k-1}^{(k-1)}) - H(X_{n+k} | X_{n+k-1}^{(k-1)}, X_{n})
        =   H(X_{n+k},X_{n+k-1}^{(k-1)}) - H(X_{n+k-1}^{(k-1)})
        - H(X_{n+k},X_{n+k-1}^{(k)}) + H(X_{n+k-1}^{(k)})
        = H(X_{n+k}^{(k)}) - H(X_{n+k-1}^{(k-1)}) - H(X_{n+k}^{(k+1)}) + H(X_{n+k-1}^{(k)})

    Parameters
    ----------
    %(segmentation)s
    lags : int | list, tuple, array of shape ``(n_lags,)``
        The lags at which to compute the auto-information function.
        If int, will use ``lags = np.arange(lags)``.
    %(ignore_repetitions)s
    %(log_base)s
    %(n_jobs)s

    Returns
    -------
    lags : array of shape ``(n_lags,)``
        Lag values in sample.
    pai : array of shape ``(n_lags,)``
        Partial auto-information array for each lag.

    References
    ----------
    .. footbibliography::
    """  # noqa: E501
    labels = _check_segmentation(segmentation)
    lags = _check_lags(lags)
    _check_type(ignore_repetitions, (bool,), "ignore_repetitions")
    log_base = _check_log_base(log_base)
    n_jobs = _check_n_jobs(n_jobs)
    if ignore_repetitions:  # ignore transition to itself (i.e. 1 -> 1)
        labels = np.array([s for s, _ in itertools.groupby(labels)])
    parallel, p_fun, _ = parallel_func(
        _partial_auto_information, n_jobs, total=len(lags)
    )
    runs = parallel(
        p_fun(labels, k, state_to_ignore=-1, log_base=log_base) for k in lags
    )
    pai = np.array(runs)
    return lags, pai
