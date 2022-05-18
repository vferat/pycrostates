from pathlib import Path

import mne
import pytest
from mne.datasets import testing

from pycrostates.io import ChData
from pycrostates.preprocessing import resample

dir_ = Path(testing.data_path()) / "MEG" / "sample"
fname_raw_testing = dir_ / "sample_audvis_trunc_raw.fif"
raw = mne.io.read_raw_fif(fname_raw_testing, preload=False)
raw = raw.pick("eeg")
raw.load_data()
epochs = mne.make_fixed_length_epochs(raw, duration=2, preload=True)


# pylint: disable=protected-access
@pytest.mark.parametrize(
    "inst, replace",
    [
        (raw, True),
        (raw, False),
        (epochs, True),
        (epochs, False),
    ],
)
def test_resample_n_epochs_n_samples(inst, replace):
    resamples = resample(inst, n_epochs=10, n_samples=500, replace=replace)
    assert isinstance(resamples, list)
    assert len(resamples) == 10
    for r in resamples:
        assert isinstance(r, ChData)
        assert r._data.shape == (59, 500)


@pytest.mark.parametrize(
    "inst, replace",
    [
        (raw, True),
        (raw, False),
        (epochs, True),
        (epochs, False),
    ],
)
def test_resample_n_epochs_coverage(inst, replace):
    resamples = resample(inst, n_epochs=10, coverage=0.8, replace=replace)
    assert isinstance(resamples, list)
    assert len(resamples) == 10
    for r in resamples:
        assert isinstance(r, ChData)


@pytest.mark.parametrize(
    "inst, replace",
    [
        (raw, True),
        (raw, False),
        (epochs, True),
        (epochs, False),
    ],
)
def test_resample_n_samples_coverage(inst, replace):
    resamples = resample(inst, n_samples=500, coverage=0.8, replace=replace)
    assert isinstance(resamples, list)
    for r in resamples:
        assert isinstance(r, ChData)
        assert r._data.shape == (59, 500)


def test_resample_raw_noreplace_error():
    try:
        resample(raw, n_epochs=1000, n_samples=5000, replace=False)
    except Exception as e:
        assert isinstance(e, ValueError)
