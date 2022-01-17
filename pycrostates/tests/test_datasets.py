import os
import mne
from pycrostates.datasets import lemon


def test_get_config():
    file_path = lemon.load_data(subject='010003', condition='EC')
    assert os.path.isfile(file_path)
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    assert isinstance(raw, mne.io.eeglab.eeglab.RawEEGLAB)
