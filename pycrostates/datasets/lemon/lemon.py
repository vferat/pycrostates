"""Functions to use the LEMON dataset."""

try:
    from importlib.resources import files  # type: ignore
except ImportError:
    from importlib_resources import files  # type: ignore

from pathlib import Path

import numpy as np
import pooch
from mne import create_info
from mne.io import BaseRaw, RawArray

from ...utils._checks import _check_type, _check_value
from ...utils._config import get_config


def data_path(subject_id: str, condition: str) -> Path:
    """Get path to a local copy of preprocessed EEG recording from the LEMON dataset.

    Get path to a local copy of preprocessed EEG recording from the
    mind-brain-body dataset of MRI, EEG, cognition, emotion, and peripheral
    physiology in young and old adults.
    If there is no local copy of the recording, this function will fetch it
    from the online repository and store it. The default location is
    ``~/pycrostates_data``.

    Parameters
    ----------
    subject_id : str
        The subject id to use. For example ``'010276'``.
        The list of available subjects can be found
        at https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID.
    condition : str
        Can be ``'EO'`` for eyes open condition or ``'EC'`` for eyes closed
        condition.

    Returns
    -------
    path : Path
        Path to a local copy of the requested recording.

    References
    ----------
    .. [1] Babayan, A., Erbey, M., Kumral, D. et al.
           A mind-brain-body dataset of MRI, EEG, cognition, emotion,
           and peripheral physiology in young and old adults. Sci Data 6, 180308 (2019).
           https://doi.org/10.1038/sdata.2018.308

    Notes
    -----
    The lemon datasets is composed of EEGLAB files. To use the MNE reader
    :func:`mne.io.read_raw_eeglab`, the ``pymatreader`` optional dependency
    is required. Use the following installation method appropriate for your
    environment:

    - ``pip install pymatreader``
    - ``conda install -c conda-forge pymatreader``

    Note that an environment created via the MNE installers includes
    ``pymatreader`` by default.
    """  # noqa: E501
    _check_type(subject_id, (str,), "subject_id")
    _check_type(condition, (str,), "condition")
    _check_value(condition, ("EO", "EC"), "condition")

    config = get_config()
    fetcher = pooch.create(
        path=config["PREPROCESSED_LEMON_DATASET_PATH"],
        base_url="https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/EEG_MPILMBB_LEMON/EEG_Preprocessed_BIDS_ID/EEG_Preprocessed/",  # noqa,
        version=None,
        registry=None,
    )
    registry = str(
        files("pycrostates.datasets.lemon").joinpath(
            "data/PREPROCESSED_LEMON_registry.txt"
        )
    )
    fetcher.load_registry(registry)

    filename_set = f"sub-{subject_id}_{condition}.set"
    filename_fdt = f"sub-{subject_id}_{condition}.fdt"
    path = fetcher.fetch(filename_set)
    _ = fetcher.fetch(filename_fdt)
    return Path(path)


def standardize(raw: BaseRaw):
    """Standardize :class:`~mne.io.Raw` from the lemon dataset.

    This function will interpolate missing channels from the standard setup,
    then reorder channels and finally reference to a common average.

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
    :func:`mne.channels.equalize_channels` instead to have the same electrodes
    across different recordings.
    """
    raw = raw.copy()
    # fmt: off
    standard_channels = [
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5",
        "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
        "CP5", "CP1", "CP2", "CP6", "AFz", "P7", "P3", "Pz",
        "P4", "P8", "PO9", "O1", "Oz", "O2", "PO10", "AF7",
        "AF3", "AF4", "AF8", "F5", "F1", "F2", "F6", "FT7",
        "FC3", "FC4", "FT8", "C5", "C1", "C2", "C6", "TP7",
        "CP3", "CPz", "CP4", "TP8", "P5", "P1", "P2", "P6",
        "PO7", "PO3", "POz", "PO4", "PO8",
    ]
    # fmt: on
    missing_channels = list(set(standard_channels) - set(raw.info["ch_names"]))

    if len(missing_channels) != 0:
        # add the missing channels as bads (array of zeros)
        missing_data = np.zeros((len(missing_channels), raw.n_times))
        data = np.vstack([raw.get_data(), missing_data])
        ch_names = raw.info["ch_names"] + missing_channels
        ch_types = raw.get_channel_types() + ["eeg"] * len(missing_channels)
        info = create_info(
            ch_names=ch_names, ch_types=ch_types, sfreq=raw.info["sfreq"]
        )
        raw = RawArray(data=data, info=info)
        raw.info["bads"].extend(missing_channels)

    raw.add_reference_channels("FCz")
    raw.reorder_channels(standard_channels)
    raw.set_montage("standard_1005")
    raw.interpolate_bads()
    raw.set_eeg_reference("average")
    return raw
