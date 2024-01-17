from pathlib import Path

from mne.io import BaseRaw

from ...utils._checks import _check_type as _check_type
from ...utils._checks import _check_value as _check_value
from ...utils._config import get_config as get_config

def data_path(subject_id: str, condition: str) -> Path:
    """Get path to a local copy of preprocessed EEG recording from the LEMON dataset.

    Get path to a local copy of preprocessed EEG recording from the mind-brain-body
    dataset of MRI, EEG, cognition, emotion, and peripheral physiology in young and old
    adults\\ :footcite:p:`babayan_mind-brain-body_2019`. If there is no local copy of the
    recording, this function will fetch it from the online repository and store it. The
    default location is ``~/pycrostates_data``.

    Parameters
    ----------
    subject_id : str
        The subject id to use. For example ``'010276'``.
        The list of available subjects can be found on this
        `FTP server <https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID>`_.
    condition : str
        Can be ``'EO'`` for eyes open condition or ``'EC'`` for eyes closed condition.

    Returns
    -------
    path : Path
        Path to a local copy of the requested recording.

    Notes
    -----
    The lemon datasets is composed of EEGLAB files. To use the MNE reader
    :func:`mne.io.read_raw_eeglab`, the ``pymatreader`` optional dependency
    is required. Use the following installation method appropriate for your
    environment:

    - ``pip install pymatreader``
    - ``conda install -c conda-forge pymatreader``

    Note that an environment created via the MNE installers includes ``pymatreader`` by
    default.

    References
    ----------
    .. footbibliography::
    """

def standardize(raw: BaseRaw):
    """Standardize :class:`~mne.io.Raw` from the lemon dataset.

    This function will interpolate missing channels from the standard setup, then
    reorder channels and finally reference to a common average.

    Parameters
    ----------
    raw : Raw
        Raw data from the lemon dataset.

    Returns
    -------
    raw : Raw
        Standardize raw.

    Notes
    -----
    If you don't want to interpolate missing channels, you can use
    :func:`mne.channels.equalize_channels` instead to have the same electrodes across
    different recordings.
    """
