from .._typing import ScalarFloatArray as ScalarFloatArray
from .._typing import ScalarIntArray as ScalarIntArray
from ..utils._checks import _check_n_jobs as _check_n_jobs
from ..utils._checks import _check_type as _check_type
from ..utils._checks import _check_value as _check_value
from ..utils._checks import _ensure_int as _ensure_int
from ..utils._docs import fill_doc as fill_doc
from ._base import _BaseSegmentation as _BaseSegmentation

def _check_log_base(log_base) -> float: ...
def _check_lags(lags) -> ScalarIntArray: ...
def _joint_entropy(
    x: ScalarIntArray,
    y: ScalarIntArray,
    state_to_ignore: int | None = -1,
    log_base: float | str = 2,
) -> float:
    """Joint Shannon entropy of the symbolic sequences x, y.

    Parameters
    ----------
    x : array (n_symbols, )
        Symbolic sequence.
    y : array (n_symbols, )
        Symbolic sequence.
    state_to_ignore : int | None
        Ignore state with symbol ``state_to_ignore`` from analysis.
    log_base : float | str
        The log base to use.
        If string:
        * ``bits``: log_base = ``2``
        * ``natural``: log_base = ``np.e``
        * ``dits``: log_base = ``10``
        Default to ``bits``.

    Returns
    -------
    h : float
        joint Shannon entropy of (x, y).
    """

def _check_labels(labels, item_name: str = "labels") -> None: ...
def _check_segmentation(
    segmentation, item_name: str = "segmentation"
) -> ScalarIntArray: ...
def _joint_entropy_history(
    labels: ScalarIntArray,
    k: int,
    state_to_ignore: int | None = -1,
    log_base: float | str = 2,
) -> float:
    """Compute the joint Shannon of k-histories x[t:t+k].

    Compute the joint Shannon entropy of the k-histories x[t:t+k].

    Parameters
    ----------
    labels : array (n_symbols, )
        Microstate symbolic sequence.
    state_to_ignore : int | None
        Ignore state with symbol ``state_to_ignore`` from analysis.
    log_base : float | str
        The log base to use.
        If string:
        * ``bits``: log_base = ``2``
        * ``natural``: log_base = ``np.e``
        * ``dits``: log_base = ``10``
        Default to ``bits``.

    Returns
    -------
    h : float
        The Shannon entropy of the sequence labels[t:t+k].
    """

def _entropy(
    labels: ScalarIntArray, state_to_ignore: int | None = -1, log_base: float | str = 2
) -> float:
    """Compute the Shannon entropy of the a symbolic sequence.

    Parameters
    ----------
    labels : array (n_symbols, )
        Microstate symbolic sequence.
    state_to_ignore : int | None
        Ignore state with symbol ``state_to_ignore`` from analysis.
    ignore_repetitions : bool
        If ``True``, ignores state repetitions.
        For example, the input sequence ``AAABBCCD``
        will be transformed into ``ABCD`` before any calculation.
        This is equivalent to setting the duration of all states to 1 sample.
    log_base : float | str
        The log base to use.
        If string:
        * ``bits``: log_base = ``2``
        * ``natural``: log_base = ``np.e``
        * ``dits``: log_base = ``10``
        Default to ``bits``.

    Returns
    -------
    h : float
        The Shannon entropy of the sequence.
    """

def entropy(
    segmentation: _BaseSegmentation,
    ignore_repetitions: bool = False,
    log_base: float | str = 2,
) -> float:
    """Compute the Shannon entropy of a symbolic sequence.

    Compute the Shannon entropy\\ :footcite:p:`shannon1948mathematical` of the microstate
    symbolic sequence.

    Parameters
    ----------
    segmentation : RawSegmentation | EpochsSegmentation
        Segmentation object containing the microstate symbolic sequence.
    ignore_repetitions : bool
        If ``True``, ignores state repetitions.
        For example, the input sequence ``AAABBCCD``
        will be transformed into ``ABCD`` before any calculation.
        This is equivalent to setting the duration of all states to 1 sample.
    log_base : float | str
        The log base to use.
        If string:
        * ``bits``: log_base = ``2``
        * ``natural``: log_base = ``np.e``
        * ``dits``: log_base = ``10``
        Default to ``bits``.

    Returns
    -------
    h : float
        The Shannon entropy of the sequence.

    References
    ----------
    .. footbibliography::
    """

def _excess_entropy_rate(
    labels: ScalarIntArray,
    history_length: int,
    state_to_ignore: int | None = -1,
    log_base: float | str = 2,
    n_jobs: int = 1,
) -> tuple[float, float, float, ScalarIntArray, ScalarFloatArray]:
    """Estimate the entropy rate and the excess_entropy from a linear fit.

    Parameters
    ----------
    labels : array (n_symbols, )
        Microstate symbolic sequence.
    history_length : int
        Maximum history length in sample to estimate the excess entropy rate.
    state_to_ignore : int | None
        Ignore state with symbol ``state_to_ignore`` from analysis.
    log_base : float | str
        The log base to use.
        If string:
        * ``bits``: log_base = ``2``
        * ``natural``: log_base = ``np.e``
        * ``dits``: log_base = ``10``
        Default to ``bits``.
    n_jobs : int | None
        The number of jobs to run in parallel. If ``-1``, it is set
        to the number of CPU cores. Requires the :mod:`joblib` package.
        ``None`` (default) is a marker for 'unset' that will be interpreted
        as ``n_jobs=1`` (sequential execution) unless the call is performed under
        a :class:`joblib:joblib.parallel_config` context manager that sets another
        value for ``n_jobs``.

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

def excess_entropy_rate(
    segmentation: _BaseSegmentation,
    history_length: int,
    ignore_repetitions: bool = False,
    log_base: float | str = 2,
    n_jobs: int = 1,
) -> tuple[float, float, float, ScalarIntArray, ScalarFloatArray]:
    """Estimate the entropy rate and the ``excess_entropy`` of the segmentation.

    The entropy rate and the ``excess_entropy`` are estimated from a linear fit:

    .. math::

        H(X_{n}^{(k)}) = a \\cdot k + b

    where ``a`` is the entropy rate and ``b`` the excess entropy
    as described in :footcite:t:`von2018partial`.

    Parameters
    ----------
    segmentation : RawSegmentation | EpochsSegmentation
        Segmentation object containing the microstate symbolic sequence.
    history_length : int
        Maximum history length in sample to estimate the excess entropy rate.
    ignore_repetitions : bool
        If ``True``, ignores state repetitions.
        For example, the input sequence ``AAABBCCD``
        will be transformed into ``ABCD`` before any calculation.
        This is equivalent to setting the duration of all states to 1 sample.
    log_base : float | str
        The log base to use.
        If string:
        * ``bits``: log_base = ``2``
        * ``natural``: log_base = ``np.e``
        * ``dits``: log_base = ``10``
        Default to ``bits``.
    n_jobs : int | None
        The number of jobs to run in parallel. If ``-1``, it is set
        to the number of CPU cores. Requires the :mod:`joblib` package.
        ``None`` (default) is a marker for 'unset' that will be interpreted
        as ``n_jobs=1`` (sequential execution) unless the call is performed under
        a :class:`joblib:joblib.parallel_config` context manager that sets another
        value for ``n_jobs``.

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

def _auto_information(
    labels: ScalarIntArray,
    k: int,
    state_to_ignore: int | None = -1,
    log_base: float | str = 2,
) -> float:
    """Compute the Auto-information for lag k.

    Parameters
    ----------
    labels : array (n_symbols, )
        Microstate symbolic sequence.
    k : int
        Lag value in sample.
    state_to_ignore : int | None
        Ignore state with symbol ``state_to_ignore`` from analysis.
    log_base : float | str
        The log base to use.
        If string:
        * ``bits``: log_base = ``2``
        * ``natural``: log_base = ``np.e``
        * ``dits``: log_base = ``10``
        Default to ``bits``.

    Returns
    -------
    a : float
        time-lagged auto-information for lag k.
    """

def auto_information_function(
    segmentation: _BaseSegmentation,
    lags: int | list[int] | tuple[int, ...] | ScalarIntArray,
    ignore_repetitions: bool = False,
    log_base: float | str = 2,
    n_jobs: int = 1,
) -> tuple[ScalarIntArray, ScalarFloatArray]:
    """Compute the Auto-information function (aif).

    Compute the Auto-information function (aif) as described
    in :footcite:t:`von2018partial`:

    .. math::

        I(X_{n+k} ; X_{n})
        = H(X_{n+k}) - H(X_{n+k} | X_{n})
        = H(X_{n+k}) - H(X_{n+k},X_{n}) + H(X_{n})
        = H(X_{n+k}) + H(X_{n}) - H(X_{n+k},X_{n})

    Parameters
    ----------
    segmentation : RawSegmentation | EpochsSegmentation
        Segmentation object containing the microstate symbolic sequence.
    lags : int | list | tuple | array of shape ``(n_lags,)``
        Lags at which to compute the auto-information function.
        If int, will use ``lags = np.arange(lags)``.
    ignore_repetitions : bool
        If ``True``, ignores state repetitions.
        For example, the input sequence ``AAABBCCD``
        will be transformed into ``ABCD`` before any calculation.
        This is equivalent to setting the duration of all states to 1 sample.
    log_base : float | str
        The log base to use.
        If string:
        * ``bits``: log_base = ``2``
        * ``natural``: log_base = ``np.e``
        * ``dits``: log_base = ``10``
        Default to ``bits``.
    n_jobs : int | None
        The number of jobs to run in parallel. If ``-1``, it is set
        to the number of CPU cores. Requires the :mod:`joblib` package.
        ``None`` (default) is a marker for 'unset' that will be interpreted
        as ``n_jobs=1`` (sequential execution) unless the call is performed under
        a :class:`joblib:joblib.parallel_config` context manager that sets another
        value for ``n_jobs``.

    Returns
    -------
    lags : array of shape ``(n_lags,)``
        Lag values in sample.
    ai : array of shape ``(n_lags,)``
        Time-lagged mutual information array for each lag.

    References
    ----------
    .. footbibliography::
    """

def _partial_auto_information(
    labels: ScalarIntArray,
    k: int,
    state_to_ignore: int | None = -1,
    log_base: float | str = 2,
) -> float:
    """Compute the partial auto-information for lag k.

    Parameters
    ----------
    labels : array (n_symbols, )
        Microstate symbolic sequence.
    k : int
        Lag values in sample.
    state_to_ignore : int | None
        Ignore state with symbol ``state_to_ignore`` from analysis.
    log_base : float | str
        The log base to use.
        If string:
        * ``bits``: log_base = ``2``
        * ``natural``: log_base = ``np.e``
        * ``dits``: log_base = ``10``
        Default to ``bits``.

    Returns
    -------
    p : float
        Partial auto-information for lag k.
    """

def partial_auto_information_function(
    segmentation: _BaseSegmentation,
    lags: int | list[int] | tuple[int, ...] | ScalarIntArray,
    ignore_repetitions: bool = False,
    log_base: float | str = 2,
    n_jobs: int | None = 1,
) -> tuple[ScalarIntArray, ScalarFloatArray]:
    """Compute the Partial auto-information function.

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
    segmentation : RawSegmentation | EpochsSegmentation
        Segmentation object containing the microstate symbolic sequence.
    lags : int | list, tuple, array of shape ``(n_lags,)``
        The lags at which to compute the auto-information function.
        If int, will use ``lags = np.arange(lags)``.
    ignore_repetitions : bool
        If ``True``, ignores state repetitions.
        For example, the input sequence ``AAABBCCD``
        will be transformed into ``ABCD`` before any calculation.
        This is equivalent to setting the duration of all states to 1 sample.
    log_base : float | str
        The log base to use.
        If string:
        * ``bits``: log_base = ``2``
        * ``natural``: log_base = ``np.e``
        * ``dits``: log_base = ``10``
        Default to ``bits``.
    n_jobs : int | None
        The number of jobs to run in parallel. If ``-1``, it is set
        to the number of CPU cores. Requires the :mod:`joblib` package.
        ``None`` (default) is a marker for 'unset' that will be interpreted
        as ``n_jobs=1`` (sequential execution) unless the call is performed under
        a :class:`joblib:joblib.parallel_config` context manager that sets another
        value for ``n_jobs``.

    Returns
    -------
    lags : array of shape ``(n_lags,)``
        Lag values in sample.
    pai : array of shape ``(n_lags,)``
        Partial auto-information array for each lag.

    References
    ----------
    .. footbibliography::
    """
