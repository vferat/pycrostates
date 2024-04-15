from _typeshed import Incomplete
from mne import BaseEpochs
from mne.io import BaseRaw

from .._typing import Picks as Picks
from .._typing import ScalarFloatArray as ScalarFloatArray
from ..io import ChData as ChData
from ..utils._checks import _check_picks_uniqueness as _check_picks_uniqueness
from ..utils._checks import _check_reject_by_annotation as _check_reject_by_annotation
from ..utils._checks import _check_tmin_tmax as _check_tmin_tmax
from ..utils._checks import _check_type as _check_type
from ..utils._docs import fill_doc as fill_doc
from ..utils._logs import logger as logger

def extract_gfp_peaks(
    inst: BaseRaw | BaseEpochs,
    picks: Picks = "eeg",
    return_all: bool = False,
    min_peak_distance: int = 1,
    tmin: float | None = None,
    tmax: float | None = None,
    reject_by_annotation: bool = True,
    verbose: Incomplete | None = None,
) -> ChData:
    """:term:`Global Field Power` (:term:`GFP`) peaks extraction.

    Extract :term:`Global Field Power` (:term:`GFP`) peaks from :class:`~mne.Epochs` or
    :class:`~mne.io.Raw`.

    Parameters
    ----------
    inst : Raw | Epochs
        Instance from which to extract :term:`global field power` (GFP) peaks.
    picks : str | list | slice | None
        Channels to use for GFP computation. Note that all channels selected must have
        the same type. Slices and lists of integers will be interpreted as channel
        indices. In lists, channel name strings (e.g. ``['Fp1', 'Fp2']``) will pick the
        given channels. Can also be the string values ``“all”`` to pick all channels, or
        ``“data”`` to pick data channels. ``"eeg"`` (default) will pick all eeg
        channels. Note that channels in ``info['bads']`` will be included if their
        names or indices are explicitly provided.
    return_all : bool
        If True, the returned `~pycrostates.io.ChData` instance will include all
        channels. If False (default), the returned `~pycrostates.io.ChData` instance
        will only include channels used for GFP computation (i.e ``picks``).
    min_peak_distance : int
        Required minimal horizontal distance (``≥ 1`) in samples between neighboring
        peaks. Smaller peaks are removed first until the condition is fulfilled for all
        remaining peaks. Default to ``1``.
    tmin : float
        Start time of the raw data to use in seconds (must be >= 0).
    tmax : float
        End time of the raw data to use in seconds (cannot exceed data duration).
    reject_by_annotation : bool
        Whether to omit bad segments from the data before fitting. If ``True``
        (default), annotated segments whose description begins with ``'bad'`` are
        omitted. If ``False``, no rejection based on annotations is performed.

        Has no effect if ``inst`` is not a :class:`mne.io.Raw` object.
    verbose : int | str | bool | None
        Sets the verbosity level. The verbosity increases gradually between ``"CRITICAL"``,
        ``"ERROR"``, ``"WARNING"``, ``"INFO"`` and ``"DEBUG"``. If None is provided, the
        verbosity is set to ``"WARNING"``. If a bool is provided, the verbosity is set to
        ``"WARNING"`` for False and to ``"INFO"`` for True.

    Returns
    -------
    ch_data : ChData
        Samples at global field power peaks.

    Notes
    -----
    The :term:`Global Field Power` (:term:`GFP`) peaks are extracted with
    :func:`scipy.signal.find_peaks`. Only the ``distance`` argument is filled with the
    value provided in ``min_peak_distance``. The other arguments are set to their
    default values.
    """

def _extract_gfp_peaks(
    data: ScalarFloatArray, min_peak_distance: int = 2
) -> ScalarFloatArray:
    """Extract GFP peaks from input data.

    Parameters
    ----------
    data : array of shape (n_channels, n_samples)
        The data to extract GFP peaks from.
    min_peak_distance : int
        Required minimal horizontal distance (>= 1) in samples between neighboring
        peaks. Smaller peaks are removed first until the condition is fulfilled for all
        remaining peaks. Default to 2.

    Returns
    -------
    peaks : array of shape (n_picks,)
        The indices when peaks occur.
    """
