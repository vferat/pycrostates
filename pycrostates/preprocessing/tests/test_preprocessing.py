from pathlib import Path

import mne
import pytest
from mne.datasets import testing

from pycrostates.io import ChData
from pycrostates.preprocessing import extract_gfp_peaks, resample

dir_ = Path(testing.data_path()) / "MEG" / "sample"
fname_raw_testing = dir_ / "sample_audvis_trunc_raw.fif"
raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
raw = raw.pick("eeg")
epochs = mne.make_fixed_length_epochs(raw, duration=2, preload=True)


@pytest.mark.parametrize(
    "inst, replace",
    [
        (raw, True),
        (raw, False),
        (epochs, True),
        (epochs, False),
    ],
)
def test_resample(inst, replace):
    ch_data = resample(inst, 10, 500, replace=replace)
    assert ch_data._data.shape == (10, 60, 500)


@pytest.mark.parametrize(
    "inst",
    [
        (raw),
        (epochs),
    ],
)
def test_resample_coverage(inst):
    ch_data = resample(inst, n_epochs=10, coverage=0.8, replace=False)
    assert len(ch_data._data) == 10


@pytest.mark.parametrize(
    "inst",
    [
        (raw),
        (epochs),
    ],
)
def test_resample_raw_samples_coverage(inst):
    ch_data = resample(inst, n_samples=500, coverage=0.8, replace=False)
    assert ch_data._data.shape[-1] == 500


def test_resample_raw_noreplace_error():
    try:
        resample(raw, 1000, 5000, replace=False)
    except Exception as e:
        assert isinstance(e, ValueError)


@pytest.mark.parametrize(
    "inst",
    [
        (raw),
        (epochs),
    ],
)
def test_extract_gfp(inst):
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick("eeg")
    ch_data = extract_gfp_peaks(inst, min_peak_distance=4)
    assert isinstance(ch_data, ChData)
