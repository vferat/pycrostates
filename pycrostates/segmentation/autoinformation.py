"""Partial autoinformation module for segmented data."""

# code from https://github.com/Frederic-vW/AIF-PAIF:
# F. von Wegner, Partial Autoinformation to Characterize Symbolic Sequences
# Front Physiol (2018) https://doi.org/10.3389/fphys.2018.01382
import numpy as np
import scipy.stats
from mne.parallel import parallel_func
from numpy.typing import NDArray

from ..utils._checks import _check_n_jobs, _check_type, _check_value
from ..utils._docs import fill_doc

# TODO: Adapt with unlabeled datapoints.
# TODO: should we use n_clusters or np.sum(p>0) ?
# TODO: should we normalize aif and paif?
# The n_clusters parameter should reflect the total number of possible states,
# including the ignored state, if applicable. If the ignored state is not
# considered as a valid state, it should be excluded from the count of
# n_clusters.

#  the Shannon entropy is not affected by non-appearing states


def _check_log_base(log_base):
    _check_type(
        log_base,
        (
            "numeric",
            str,
        ),
        "log_base",
    )
    if isinstance(log_base, str):
        _check_value(
            log_base,
            ["bits", "natural", "dits"],
            item_name="log_base",
            extra="when string is provided",
        )
        if log_base == "bits":
            log_base = 2
        elif log_base == "natural":
            log_base = np.e
        elif log_base == "dits":
            log_base = 10
    else:
        if log_base <= 0:
            raise ValueError("If numeric, log_base must be >= 0.")
    return log_base


@fill_doc
def joint_entropy(
    x: NDArray[int], y: NDArray[int], state_to_ignore=-1, log_base: float = 2
):
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
    _check_type(state_to_ignore, (int,), "state_to_ignore")
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


@fill_doc
def joint_entropy_history(
    labels: NDArray[int], k: int, state_to_ignore=-1, log_base: float = 2
):
    r"""Compute the joint Shannon of k-histories x[t:t+k].

    Compute the joint Shannon entropy
    \ :footcite:p:`shannon1948mathematicalof`..
    of the k-histories x[t:t+k].

    Parameters
    ----------
    %(labels_info)s
    %(state_to_ignore)s
    %(log_base)s

    Returns
    -------
    h : float
        The Shannon entropy of the sequence labels[t:t+k].

    References
    ----------
    .. footbibliography::
    """
    _check_type(labels, (np.ndarray,), "labels")
    _check_type(k, ("int",), "k")
    _check_type(state_to_ignore, (int,), "state_to_ignore")
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
    for i in range(
        len(labels) - k + 1
    ):  # TODO: check +1 (not in original code)
        history = tuple(labels[i : i + k])
        if state_to_ignore not in history:
            joint_dist[history] += 1.0
    # Compute the joint entropy
    joint_entropy = scipy.stats.entropy(joint_dist.flatten(), base=log_base)
    return joint_entropy


@fill_doc
def entropy(
    labels: NDArray[int], state_to_ignore: int = -1, log_base: float = 2
):
    r"""Compute the Shannon entropy of the a symbolic sequence.

    Compute the Shannon entropy
    \ :footcite:p:`shannon1948mathematicalof`..
    of the a symbolic sequence.

    Parameters
    ----------
    %(labels_info)s
    %(state_to_ignore)s
    %(log_base)s

    Returns
    -------
    h : float
        The Shannon entropy of the sequence.
    """
    _check_type(labels, (np.ndarray,), "x")
    _check_type(state_to_ignore, (int,), "state_to_ignore")
    log_base = _check_log_base(log_base)

    h = joint_entropy_history(
        labels, k=1, state_to_ignore=state_to_ignore, log_base=log_base
    )
    return h


@fill_doc
def ais(
    labels: NDArray[int],
    k: int,
    state_to_ignore=-1,
    log_base: float = 2,
):
    r"""
    Compute the Active information storage (AIS).

    Compute the Active information storage (AIS) as
    described in \ :footcite:p:`von2018partial`..

    TeX notation:
    I(X_{n+1} ; X_{n}^{(k)})
    = H(X_{n+1}) - H(X_{n+1} | X_{n}^{(k)})
    = H(X_{n+1}) - H(X_{n+1},X_{n}^{(k)}) + H(X_{n}^{(k)})
    = H(X_{n+1}) + H(X_{n}^{(k)}) - H(X_{n+1}^{(k+1)})

    Parameters
    ----------
    %(labels_info)s
    %(k_info)s
    %(state_to_ignore)s
    %(log_base)s

    Returns
    -------
    a: float
       The active information storage.

    References
    ----------
    .. footbibliography::
    """
    _check_type(labels, (np.ndarray,), "labels")
    _check_type(k, (int,), "k")
    if k == 1:
        raise ValueError(
            "The joint Shannon entropy of k-histories"
            "is not directly applicable for k=1."
        )
    _check_type(state_to_ignore, (int,), "state_to_ignore")
    log_base = _check_log_base(log_base)

    h1 = entropy(
        labels, state_to_ignore=state_to_ignore
    )  # TODO: joint_entropy_history(k=1 != entropy ?)
    h2 = joint_entropy_history(
        labels, k, state_to_ignore=state_to_ignore, log_base=log_base
    )
    h3 = joint_entropy_history(
        labels, k + 1, state_to_ignore=state_to_ignore, log_base=log_base
    )
    a = h1 + h2 - h3
    return a


# ---- TODO ---- #
@fill_doc
def aif(
    labels: NDArray[int],
    k: int,
    state_to_ignore=-1,
    log_base: float = 2,
):
    """
    Time-lagged mutual information = Auto-information function (AIF)

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

    Returns
    -------
    a: float
        time-lagged mutual information array for lag k.
    """
    _check_type(labels, (np.ndarray,), "labels")
    _check_type(k, (int,), "k")
    if k == 1:
        raise ValueError(
            "The joint Shannon entropy of k-histories"
            "is not directly applicable for k=1."
        )
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
    h12 = joint_entropy(
        labels[:nmax],
        labels[k : k + nmax],
        state_to_ignore=state_to_ignore,
        log_base=log_base,
    )
    a = h1 + h2 - h12
    return a


@fill_doc
def aif_p(
    labels: NDArray[int],
    ks: int,
    list,
    state_to_ignore=-1,
    log_base: float = 2,
    n_jobs: int = None,
):
    """
    Time-lagged mutual information = Auto-information function (AIF)

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
    a: float (n_symbols, )
        time-lagged mutual information array for lags k=0, ..., kmax-1
    """
    _check_type(labels, (np.ndarray,), "labels")
    _check_type(k, (int,), "k")
    if ks == 1:
        raise ValueError(
            "The joint Shannon entropy of k-histories"
            "is not directly applicable for k=1."
        )
    _check_type(state_to_ignore, (int,), "state_to_ignore")
    log_base = _check_log_base(log_base)
    n_jobs = _check_n_jobs(n_jobs)

    if isinstance(ks, int):
        ks = list(range(ks))

    parallel, p_fun, _ = parallel_func(aif, n_jobs, total=len(ks))
    runs = parallel(
        p_fun(labels, k, state_to_ignore=state_to_ignore, log_base=log_base)
        for k in ks
    )
    return runs


@fill_doc
def paif(
    labels: NDArray[int],
    n_clusters: int,
    kmax: int,
    bias_correction: bool = False,
    log_base: float = 2,
):
    """
    Partial auto-information function (PAIF).

    TeX notation:
    I(X_{n+k} ; X_{n} | X_{n+k-1}^{(k-1)})
    = H(X_{n+k} | X_{n+k-1}^{(k-1)}) - H(X_{n+k} | X_{n+k-1}^{(k-1)}, X_{n})
    =   H(X_{n+k},X_{n+k-1}^{(k-1)}) - H(X_{n+k-1}^{(k-1)})
      - H(X_{n+k},X_{n+k-1}^{(k)}) + H(X_{n+k-1}^{(k)})
    = H(X_{n+k}^{(k)}) - H(X_{n+k-1}^{(k-1)}) - H(X_{n+k}^{(k+1)}) + H(X_{n+k-1}^{(k)})

    Parameters
    ----------
    %(labels_info)s
    kmax: int
        Maximum time lag;
    %(log_base)s

    Returns
    -------
    p: array (n_symbols, )
        Partial auto-information array for lags k=0, ..., kmax-1
    """

    def p_function(
        labels,
        n_clusters,
        k,
        bias_correction=bias_correction,
        log_base=log_base,
    ):
        h1 = H_k(
            labels,
            n_clusters,
            k,
            bias_correction=bias_correction,
            log_base=log_base,
        )
        h2 = H_k(
            labels,
            n_clusters,
            k - 1,
            bias_correction=bias_correction,
            log_base=log_base,
        )
        h3 = H_k(
            labels,
            n_clusters,
            k + 1,
            bias_correction=bias_correction,
            log_base=log_base,
        )
        p = 2 * h1 - h2 - h3
        return p

    p = np.zeros(kmax)
    a = aif(labels, n_clusters, 2)  # use AIF coeffs. for k=0, 1
    p[0], p[1] = a[0], a[1]
    for k in range(2, kmax):
        p[k] = p_function(
            labels,
            n_clusters,
            k,
            bias_correction=bias_correction,
            log_base=log_base,
        )
    p /= p[0]  # normalize: p[0]=1.0
    return p
