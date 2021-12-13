import os.path as op

import mne
from mne.io.array.array import RawArray
from mne.datasets import testing
from pycrostates.preprocessing import resample, extract_gfp_peaks

data_path = testing.data_path()
fname_raw_testing = op.join(data_path, 'MEG', 'sample',
                            'sample_audvis_trunc_raw.fif')

raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
raw = raw.pick('eeg')
epochs = mne.make_fixed_length_epochs(raw, duration=2, preload=True)


def test_resample_raw_replace():
    r = resample(raw, 10, 5000)
    assert len(r) == 10
    assert r[0].n_times == 5000


def test_resample_epochs_replace():
    r = resample(epochs, 10, 5000)
    assert len(r) == 10
    assert r[0].n_times == 5000


def test_resample_raw_noreplace():
    r = resample(raw, 10, 500, replace=False)
    assert len(r) == 10
    assert r[0].n_times == 500


def test_resample_raw_epochs_coverage():
    r = resample(raw, n_epochs=10, coverage=0.8, replace=False)
    assert len(r) == 10


def test_resample_raw_samples_coverage():
    r = resample(raw, n_samples=500, coverage=0.8, replace=False)
    assert r[0].n_times == 500


def test_resample_raw_noreplace_error():
    try:
        resample(raw, 1000, 5000, replace=False)
    except Exception as e:
        assert isinstance(e, ValueError)


def test_extract_gfp_raw():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    raw_peaks = extract_gfp_peaks(raw, min_peak_distance=4)
    assert isinstance(raw_peaks, RawArray)
    assert(raw_peaks.info['sfreq'] == -1)


def test_extract_gfp_epochs():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    events = mne.make_fixed_length_events(raw, 1)
    epochs = mne.epochs.Epochs(raw, events, preload=True)
    epochs = epochs.pick('eeg')
    raw_peaks = extract_gfp_peaks(epochs, min_peak_distance=4)
    assert isinstance(raw_peaks, RawArray)
