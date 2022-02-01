
from pathlib import Path

import numpy as np
import mne
from mne.datasets import testing

from pycrostates.utils import get_config, _copy_info

dir_ = Path(testing.data_path()) / 'MEG' / 'sample'
fname_raw_testing = dir_ / 'sample_audvis_trunc_raw.fif'
raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
raw = raw.pick('eeg')
raw = raw.filter(0, 40)
raw = raw.crop(0, 10)


def test_get_config():
    config = get_config()
    assert isinstance(config, dict)


def test_copy_info():
    info = _copy_info(raw, sfreq=np.inf)
    assert info['sfreq'] == np.inf

