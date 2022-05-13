from pathlib import Path

import mne
import pytest
from mne.datasets import testing

from pycrostates.io import ChData
from pycrostates.preprocessing import extract_gfp_peaks


dir_ = Path(testing.data_path()) / "MEG" / "sample"
fname_raw_testing = dir_ / "sample_audvis_trunc_raw.fif"
raw = mne.io.read_raw_fif(fname_raw_testing, preload=False)
raw = raw.pick("eeg")
raw.load_data()
epochs = mne.make_fixed_length_epochs(raw, duration=2, preload=True)


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
