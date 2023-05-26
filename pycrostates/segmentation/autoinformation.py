"""Partial autoinformation module for segmented data."""

# code from https://github.com/Frederic-vW/AIF-PAIF:
# F. von Wegner, Partial Autoinformation to Characterize Symbolic Sequences
# Front Physiol (2018) https://doi.org/10.3389/fphys.2018.01382

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ..utils._checks import _check_type

# --- define logarithm
# log = np.log
log = np.log2

# TODO: Adapt with unlabeled datapoints.


def H_1(labels: NDArray[int], n_clusters: int, log_base: float = 2):
    """Compute the Shannon entropy of the a symbolic sequence.

    Parameters
    ----------
    labels : array (n_symbols, )
        Symbolic sequence.
    n_clusters: int
        Number of unique symbols.
    log_base : float
        The log base to use, Default to 2 (gives the unit of bits).

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
    h = -np.sum(p[p > 0] * (log(p[p > 0]) / np.log(log_base)))
    return h


def ais(labels, n_clusters, k=1):
    """
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
        x: symbolic sequence, symbols = [0, 1, ..., ns-1]
        ns: number of symbols
        k: history length (optional, default value k=1)

    Returns
    -------
        a: active information storage
        
    References
    ----------
    .. footbibliography::
    """
    _check_type(labels, (np.ndarray,), "labels")
    _check_type(n_clusters, (int,), "n_clusters")
    _check_type(log_base, (float,), "log_base")
    n = len(labels) # TODO: from original code unused or error ?
    h1 = H_k(labels, n_clusters, 1)
    h2 = H_k(labels, n_clusters, k)
    h3 = H_k(labels, n_clusters, k + 1)
    a = h1 + h2 - h3
    return a


def aif(x, ns, kmax):
    """
    Time-lagged mutual information = Auto-information function (AIF)

    TeX notation:
    I(X_{n+k} ; X_{n})
    = H(X_{n+k}) - H(X_{n+k} | X_{n})
    = H(X_{n+k}) - H(X_{n+k},X_{n}) + H(X_{n})
    = H(X_{n+k}) + H(X_{n}) - H(X_{n+k},X_{n})

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
        kmax: maximum time lag
    Returns:
        a: time-lagged mutual information array for lags k=0, ..., kmax-1
    """

    n = len(x)
    a = np.zeros(kmax)
    for k in range(kmax):
        nmax = n - k
        h1 = H_1(x[:nmax], ns)
        h2 = H_1(x[k : k + nmax], ns)
        h12 = H_2(x[:nmax], x[k : k + nmax], ns)
        a[k] = h1 + h2 - h12
        # if (k%10 == 0): print(f"\tAIF(k={k:d})\r", end="")
        print(f"\tAIF(k={k:d})\r", end="")
    print("")
    a /= a[0]  # normalize: a[0]=1.0
    return a


def paif(x, ns, kmax):
    """
    Partial auto-information function (PAIF).

    TeX notation:
    I(X_{n+k} ; X_{n} | X_{n+k-1}^{(k-1)})
    = H(X_{n+k} | X_{n+k-1}^{(k-1)}) - H(X_{n+k} | X_{n+k-1}^{(k-1)}, X_{n})
    =   H(X_{n+k},X_{n+k-1}^{(k-1)}) - H(X_{n+k-1}^{(k-1)})
      - H(X_{n+k},X_{n+k-1}^{(k)}) + H(X_{n+k-1}^{(k)})
    = H(X_{n+k}^{(k)}) - H(X_{n+k-1}^{(k-1)}) - H(X_{n+k}^{(k+1)}) + H(X_{n+k-1}^{(k)})

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
        kmax: maximum time lag
    Returns:
        p: partial autoinformation array for lags k=0, ..., kmax-1
    """

    n = len(x)
    p = np.zeros(kmax)
    a = aif(x, ns, 2)  # use AIF coeffs. for k=0, 1
    p[0], p[1] = a[0], a[1]
    for k in range(2, kmax):
        h1 = H_k(x, ns, k)
        h2 = H_k(x, ns, k - 1)
        h3 = H_k(x, ns, k + 1)
        p[k] = 2 * h1 - h2 - h3
        # if (k%5 == 0): print(f"\tPAIF(k={k:d})\r", end="")
        print(f"\tPAIF(k={k:d})\r", end="")
    print("")
    p /= p[0]  # normalize: p[0]=1.0
    return p


def excess_entropy_rate(x, ns, kmax, doplot=False):
    """
    Estimate the entropy rate and the excess entropy from a linear fit:
    H(X_{n}^{(k)}) = a * k + b (history length k vs. joint entropy H_k)

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
        kmax: maximum time lag
    Returns:
        a: entropy rate (slope)
        b: excess entropy (intercept)
    """

    h = np.zeros(kmax)
    for k in range(kmax):
        h[k] = H_k(x, ns, k + 1)  # joint entropy for history length k+1
    ks = np.arange(1, kmax + 1)  # history lengths
    a, b = np.polyfit(ks, h, 1)
    # --- Figure ---
    if doplot:
        plt.figure(figsize=(6, 6))
        plt.plot(ks, h, "-sk")
        plt.plot(ks, a * ks + b, "-b")
        plt.tight_layout()
        plt.title("Entropy rate & excess entropy")
        plt.show()
    return (a, b)


def Tk_hat(x, ns, k=1):
    """
    For a symbolic sequence x, estimate the
    * joint probability distribution for history length k: pk = P(X_{n}^{(k)})
    * k-history based transition (probability) matrix: Tk = P(X_{n+1} | X_{n}^{(k)})

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
        k: history length
    Returns:
        pk: joint distribution P(X_{n}^{(k)})
        Tk: transition matrix P(X_{n+1} | X_{n}^{(k)})
    """

    n = len(x)  # -k
    L = ns**k  # number of possible k-histories
    pk = np.zeros(L)
    Tk = np.zeros((L, ns))
    # --- map k-dim. histories to 1D indices ---
    sh = tuple(k * [ns])  # shape of history array (ns...ns)
    d = {}  # dictionary idx -> i
    for i, idx in enumerate(np.ndindex(sh)):
        d[idx] = i
    for t in range(n - k):
        idx = tuple(x[t : t + k])  # k-history
        i = d[idx]  # 1D index of k-history
        j = x[t + k]  # symbol following the p-history idx
        pk[i] += 1.0
        Tk[i, j] += 1.0
    pk /= pk.sum()
    p_row = Tk.sum(axis=1, keepdims=True)
    p_row[p_row == 0] = 1.0  # avoid division by zero
    Tk /= p_row  # normalize row sums to 1.0 to obtain a stochastic matrix
    return pk, Tk


def mc_k(pk, Tk, k, ns, n):
    """
    Synthesize a surrogate k-th order Markov chain.
    pk, Tk are obtained from the function Tk_hat

    Args:
        pk: empirical symbol distribution
        Tk: empirical k-history transition matrix
        k: Markov order
        ns: number of symbols
        n: length of surrogate Markov chain
    Returns:
        x: surrogate Markov chain, symbols = [0, 1, 2, ...]
    """

    # --- map k-dim. histories to 1D indices ---
    sh = tuple(k * [ns])  # shape of history array (ns...ns)
    d = {}  # dictionary idx -> i
    dinv = np.zeros((ns**k, k))  # inverse of d
    for i, idx in enumerate(np.ndindex(sh)):
        d[idx] = i
        dinv[i] = idx

    # initialize first k-history with a random sample from pk
    pk_sum = np.cumsum(pk)
    x = np.zeros(n)
    x[:k] = dinv[np.min(np.argwhere(np.random.rand() < pk_sum))]

    # iterate transitions according to Tk, up to length n
    Tk_sum = np.cumsum(Tk, axis=1)
    for t in range(n - k - 1):
        idx = tuple(x[t : t + k])  # current k-history
        i = d[idx]  # 1D index of k-history
        j = np.min(np.argwhere(np.random.rand() < Tk_sum[i]))
        x[t + k] = j  # next sample according to Tk

    return x.astype("int")


def plot_aif_paif(x, ns, kmax):
    """
    For the symbolic sequence x with ns symbols,
    plot the AIF and PAIF coefficients up to lag kmax.
    """

    # --- AIF / PAIF ---
    print("[+] AIF:")
    aif_ = aif(x, ns, kmax)
    print("[+] PAIF:")
    paif_ = paif(x, ns, kmax)

    # --- Figure ---
    fs_tix = 14
    fs_label = 16
    fs_title = 18
    xyc = "axes fraction"
    ms_ = 5
    lw_ = 2
    lags = np.arange(kmax)
    z = np.zeros(kmax)
    ymax = 1.1

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))  # , sharex=True, sharey=True

    # --- plot AIF ---
    ax[0].plot(lags, z, "-k", lw=1)
    ax[0].plot(lags, aif_, "ok", ms=ms_)
    for i in range(len(aif_)):
        ax[0].plot([i, i], [0, aif_[i]], "-k")
    ax[0].tick_params(axis="both", labelsize=fs_tix)
    ax[0].set_xlabel("lag", fontsize=fs_label)
    ax[0].set_ylabel(r"$\alpha_k$ [bits]", fontsize=fs_label)
    ax[0].set_ylim(-0.05, ymax)
    ax[0].set_title("AIF", fontsize=16, fontweight="bold")

    # --- plot PAIF ---
    ax[1].plot(lags, z, "-k", lw=1)
    ax[1].plot(lags, paif_, "ok", ms=ms_)
    for i in range(len(paif_)):
        ax[1].plot([i, i], [0, paif_[i]], "-k")
    ax[1].tick_params(axis="both", labelsize=fs_tix)
    ax[1].set_xlabel("lag", fontsize=fs_label)
    ax[1].set_ylabel(r"$\pi_{k}$ [bits]", fontsize=fs_label)
    ax[1].set_ylim(-0.05, ymax)
    ax[1].set_title("PAIF", fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.show()
