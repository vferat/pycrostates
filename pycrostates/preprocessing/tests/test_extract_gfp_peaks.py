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


@pytest.mark.parametrize("inst", (raw, epochs))
def test_extract_gfp(inst, caplog):
    """Test valid arguments for extract_gfp_peaks."""
    ch_data = extract_gfp_peaks(inst)
    assert isinstance(ch_data, ChData)
    assert ch_data.info == inst.info
    assert "GFP peaks extracted" in caplog.text

    # with min_peak_distance
    ch_data2 = extract_gfp_peaks(inst, min_peak_distance=10)
    assert isinstance(ch_data2, ChData)
    assert ch_data2.info == inst.info
    assert ch_data != ch_data2

    # with picks
    ch_data = extract_gfp_peaks(inst, picks=inst.ch_names[0])
    assert isinstance(ch_data, ChData)
    assert ch_data.info != inst.info
    assert ch_data._data.shape[0] == 1

    # with tmin/tmax
    tmin = None
    tmax = 1
    ch_data = extract_gfp_peaks(inst, tmin=tmin, tmax=tmax)
    assert isinstance(ch_data, ChData)
    assert ch_data.info == inst.info


@pytest.mark.parametrize("inst", (raw, epochs))
def test_extract_gfp_invalid_arguments(inst):
    """Test errors raised when invalid arguments are provided."""
    with pytest.raises(TypeError, match=""):
        extract_gfp_peaks(101)
    with pytest.raises(TypeError, match=""):
        extract_gfp_peaks(inst, min_peak_distance=True)
    with pytest.raises(
        ValueError, match="Argument 'min_peak_distance' must  be "
    ):
        extract_gfp_peaks(inst, min_peak_distance=-2)
