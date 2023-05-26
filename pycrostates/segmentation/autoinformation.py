"""Partial autoinformation module for segmented data."""

# code from https://github.com/Frederic-vW/AIF-PAIF:
# F. von Wegner, Partial Autoinformation to Characterize Symbolic Sequences
# Front Physiol (2018) https://doi.org/10.3389/fphys.2018.01382

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ..utils._checks import _check_type
from ..utils._docs import fill_doc

# TODO: Adapt with unlabeled datapoints.
# TODO: should we use n_clusters or np.sum(p>0) ?
# TODO: should we normalize aif and paif?

@fill_doc
def H_1(labels: NDArray[int], n_clusters: int, log_base: float = 2):
    r"""Compute the Shannon entropy of the a symbolic sequence.

    Compute the Shannon entropy
    \ :footcite:p:`shannon1948mathematicalof`..
    of the a symbolic sequence.

    Parameters
    ----------
    %(labels_info)s
    %(n_clusters_info)s
    %(log_base)s

    Returns
    -------
    h : float
        The Shannon entropy of the sequence.
    """
    _check_type(labels, (np.ndarray,), "labels")
    _check_type(n_clusters, (int,), "n_clusters")
    _check_type(log_base, (float,), "log_base")

    n = len(labels)
    p = np.zeros(n_clusters)  # symbol distribution
    for t in range(n):
        p[labels[t]] += 1.0
    p /= n
    h = -np.sum(p[p > 0] * (np.log(p[p > 0]) / np.log(log_base)))
    return h


@fill_doc
def H_2(
    x: NDArray[int], y: NDArray[int], n_clusters: int, log_base: float = 2
):
    """
    Joint Shannon entropy of the symbolic sequences x, y, with n_clusters symbols.

    Parameters
    ----------
    x, y : array (n_symbols, )
        Symbolic sequences.
    %(n_clusters_info)s
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
    _check_type(n_clusters, (int,), "n_clusters")
    _check_type(log_base, (float,), "log_base")

    n = len(x)
    p = np.zeros((n_clusters, n_clusters))  # joint distribution
    for t in range(n):
        p[x[t], y[t]] += 1.0
    p /= n
    h = -np.sum(p[p > 0] * np.log(p[p > 0]) / np.log(log_base))
    return h


@fill_doc
def H_k(
    labels: NDArray[int],
    n_clusters: int,
    k: int,
    bias_correction: bool = False,
    log_base: float = 2,
):
    r"""Compute the joint Shannon of k-histories x[t:t+k].

    Compute the joint Shannon entropy
    \ :footcite:p:`shannon1948mathematicalof`..
    of the k-histories x[t:t+k].

    Parameters
    ----------
    %(labels_info)s
    %(n_clusters_info)s
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
    _check_type(n_clusters, (int,), "n_clusters")
    _check_type(k, (int,), "k")
    _check_type(bias_correction, (bool,), "bias_correction")
    _check_type(log_base, (float,), "log_base")

    n = len(labels)
    p = np.zeros(tuple(k * [n_clusters]))  # symbol joint distribution
    for t in range(n - k):
        p[tuple(labels[t : t + k])] += 1.0
    p /= n - k  # normalize distribution
    h = -np.sum(p[p > 0] * np.log(p[p > 0]) / np.log(log_base))
    if bias_correction:
        b = np.sum(p > 0)  # Should we include values where p=0 ?
        h = h + (b - 1) / (2 * n)
    return h


@fill_doc
def ais(
    labels: NDArray[int],
    n_clusters: int,
    k: int,
    bias_correction: bool = False,
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
    %(n_clusters_info)s
    %(k_info)s
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
    _check_type(n_clusters, (int,), "n_clusters")
    _check_type(k, (int,), "k")
    _check_type(bias_correction, (bool,), "bias_correction")
    _check_type(log_base, (float,), "log_base")
    hs = list()
    for k_ in [1, k, k + 1]:
        h = H_k(
            labels,
            n_clusters,
            k_,
            bias_correction=bias_correction,
            log_base=log_base,
        )
        hs.append(h)
    a = np.sum(hs)
    return a


@fill_doc
def aif(
    labels: NDArray[int],
    n_clusters: int,
    kmax: int,
    bias_correction: bool = False,
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
    %(n_clusters_info)s
    kmax: int
        Maximum time lag;
    %(log_base)s

    Returns
    -------
    a: array (n_symbols, )
        time-lagged mutual information array for lags k=0, ..., kmax-1
    """

    def a_function(
        labels,
        n_clusters,
        k,
        bias_correction=bias_correction,
        log_base=log_base,
    ):
        n = len(labels)
        nmax = n - k
        h1 = H_1(
            labels[:nmax],
            n_clusters,
            bias_correction=bias_correction,
            log_base=log_base,
        )
        h2 = H_1(
            labels[k : k + nmax],
            n_clusters,
            bias_correction=bias_correction,
            log_base=log_base,
        )
        h12 = H_2(
            labels[:nmax],
            labels[k : k + nmax],
            n_clusters,
            bias_correction=bias_correction,
            log_base=log_base,
        )
        a[k] = h1 + h2 - h12

    a = np.zeros(kmax)
    for k in range(kmax):  # TODO: parallel computing.
        a[k] = a_function(
            labels,
            n_clusters,
            k,
            bias_correction=bias_correction,
            log_base=log_base,
        )
    a /= a[0]  # normalize: a[0]=1.0
    return a


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
    %(n_clusters_info)s
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
