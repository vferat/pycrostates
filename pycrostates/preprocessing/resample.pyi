from _typeshed import Incomplete
from mne import BaseEpochs
from mne.io import BaseRaw

from .._typing import Picks as Picks
from .._typing import RandomState as RandomState
from ..io import ChData as ChData
from ..utils._checks import _check_random_state as _check_random_state
from ..utils._checks import _check_reject_by_annotation as _check_reject_by_annotation
from ..utils._checks import _check_tmin_tmax as _check_tmin_tmax
from ..utils._checks import _check_type as _check_type
from ..utils._docs import fill_doc as fill_doc
from ..utils._logs import logger as logger

def resample(
    inst: BaseRaw | BaseEpochs | ChData,
    picks: Picks = None,
    tmin: float | None = None,
    tmax: float | None = None,
    reject_by_annotation: bool = True,
    n_resamples: int = None,
    n_samples: int = None,
    coverage: float = None,
    replace: bool = True,
    random_state: RandomState = None,
    verbose: Incomplete | None = None,
) -> list[ChData]:
    """Resample a recording into epochs of random samples.

    Resample :class:`~mne.io.Raw`. :class:`~mne.Epochs` or
    `~pycrostates.io.ChData` into ``n_resamples`` each containing ``n_samples``
    random samples of the original recording.

    Parameters
    ----------
    inst : Raw | Epochs | ChData
        Instance to resample.
    picks : str | array-like | slice | None
        Channels to include. Slices and lists of integers will be interpreted as
        channel indices. In lists, channel *type* strings (e.g., ``['meg',
        'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
        ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
        string values "all" to pick all channels, or "data" to pick :term:`data
        channels`. None (default) will pick all channels. Note that channels in
        ``info['bads']`` *will be included* if their names or indices are
        explicitly provided.
    tmin : float
        Start time of the raw data to use in seconds (must be >= 0).
    tmax : float
        End time of the raw data to use in seconds (cannot exceed data duration).
    reject_by_annotation : bool
        Whether to omit bad segments from the data before fitting. If ``True``
        (default), annotated segments whose description begins with ``'bad'`` are
        omitted. If ``False``, no rejection based on annotations is performed.

        Has no effect if ``inst`` is not a :class:`mne.io.Raw` object.
    n_resamples : int
        Number of resamples to draw. Each epoch can be used to fit a separate clustering
        solution. See notes for additional information.
    n_samples : int
        Length of each epoch (in samples). See notes for additional information.
    coverage : float
        Strictly positive ratio between resampling data size and size of the original
        recording. See notes for additional information.
    replace : bool
        Whether or not to allow resampling with replacement.
    random_state : None | int | instance of ~numpy.random.RandomState
        A seed for the NumPy random number generator (RNG). If ``None`` (default),
        the seed will be  obtained from the operating system
        (see  :class:`~numpy.random.RandomState` for details), meaning it will most
        likely produce different output every time this function or method is run.
        To achieve reproducible results, pass a value here to explicitly initialize
        the RNG with a defined state.
    verbose : int | str | bool | None
        Sets the verbosity level. The verbosity increases gradually between ``"CRITICAL"``,
        ``"ERROR"``, ``"WARNING"``, ``"INFO"`` and ``"DEBUG"``. If None is provided, the
        verbosity is set to ``"WARNING"``. If a bool is provided, the verbosity is set to
        ``"WARNING"`` for False and to ``"INFO"`` for True.

    Returns
    -------
    resamples : list of :class:`~pycrostates.io.ChData`
        List of resamples.

    Notes
    -----
    Only two of ``n_resamples``, ``n_samples`` and ``coverage`` parameters must be
    defined, the non-defined one will be determine at runtime by the 2 other parameters.
    """
