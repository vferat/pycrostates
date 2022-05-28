import mne

from pycrostates.datasets import lemon


def test_lemon_data_path():
    file_path = lemon.data_path(subject_id="010003", condition="EC")
    assert file_path.is_file()
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    assert isinstance(raw, mne.io.eeglab.eeglab.RawEEGLAB)


def test_lemon_standardize():
    file_path = lemon.data_path(subject_id="010003", condition="EC")
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    raw = lemon.standardize(raw)
    assert isinstance(raw, mne.io.RawArray)
