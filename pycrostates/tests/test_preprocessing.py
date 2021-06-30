import os.path as op
from pycrostates.preprocessing.preprocessing import extract_gfp_peaks
from mne.io.array.array import RawArray

import numpy as np
import mne
from mne.datasets import testing

from pycrostates.clustering import ModKMeans
from pycrostates.segmentation import RawSegmentation, EpochsSegmentation, EvokedSegmentation
from pycrostates.preprocessing import resample, extract_gfp_peaks

data_path = testing.data_path()
fname_raw_testing = op.join(data_path, 'MEG', 'sample',
                            'sample_audvis_trunc_raw.fif')
fname_evoked_testing = op.join(data_path, 'MEG', 'sample',
                               'sample_audvis-ave.fif')

def test_resample_raw_replace():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    r = resample(raw, 10, 5000)
    assert len(r) == 10
    assert r[0].n_times == 5000
    
    
def test_resample_raw_noreplace():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    r = resample(raw, 10, 500, replace=False)
    assert len(r) == 10
    assert r[0].n_times == 500

def test_resample_raw_epochs_coverage():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    r = resample(raw, n_epochs=10, coverage=0.8, replace=False)
    assert len(r) == 10

def test_resample_raw_samples_coverage():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    r = resample(raw, n_samples=500, coverage=0.8, replace=False)
    assert r[0].n_times == 500
    
def test_resample_raw_noreplace_error():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    try:
        resample(raw, 1000, 5000, replace=False)
    except Exception as e:
        assert isinstance(e, ValueError)

def test_extract_gfp_raw():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    raw_peaks = extract_gfp_peaks(raw, min_peak_distance=4)
    assert isinstance(raw_peaks, RawArray)
        
def test_extract_gfp_epochs():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    events = mne.make_fixed_length_events(raw, 1)
    epochs = mne.epochs.Epochs(raw, events, preload=True)
    epochs = epochs.pick('eeg')
    raw_peaks = extract_gfp_peaks(epochs, min_peak_distance=4)
    assert isinstance(raw_peaks, RawArray)