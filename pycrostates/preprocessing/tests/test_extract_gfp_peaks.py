import mne
import pytest
from mne import pick_info
from mne.datasets import testing
from mne.utils import check_version

if check_version("mne", "1.6"):
    from mne._fiff.pick import _picks_to_idx
else:
    from mne.io.pick import _picks_to_idx

from pycrostates.io import ChData
from pycrostates.preprocessing import extract_gfp_peaks
from pycrostates.utils._logs import logger

logger.propagate = True

dir_ = testing.data_path() / "MEG" / "sample"
fname_raw_testing = dir_ / "sample_audvis_trunc_raw.fif"
raw = mne.io.read_raw_fif(fname_raw_testing, preload=False)
raw.pick("eeg").load_data()
epochs = mne.make_fixed_length_epochs(raw, duration=2, preload=True)


@pytest.mark.parametrize("inst", (raw, epochs))
def test_extract_gfp(inst, caplog):
    """Test valid arguments for extract_gfp_peaks."""
    ch_data = extract_gfp_peaks(inst)
    assert isinstance(ch_data, ChData)
    picks = _picks_to_idx(inst.info, None)
    assert ch_data.info == pick_info(inst.info, picks)
    assert "GFP peaks extracted" in caplog.text

    # with min_peak_distance
    ch_data2 = extract_gfp_peaks(inst, min_peak_distance=10)
    assert isinstance(ch_data2, ChData)
    picks = _picks_to_idx(inst.info, None)
    assert ch_data2.info == pick_info(inst.info, picks)
    assert ch_data != ch_data2

    # with picks
    ch_data = extract_gfp_peaks(inst, picks=inst.ch_names[0])
    assert isinstance(ch_data, ChData)
    assert ch_data.info != ch_data2.info
    assert ch_data._data.shape[0] == 1

    # with return_all
    ch_data = extract_gfp_peaks(inst, picks=inst.ch_names[0], return_all=True)
    assert isinstance(ch_data, ChData)
    assert ch_data.info["ch_names"] == inst.ch_names

    # with tmin/tmax
    tmin = None
    tmax = 1
    ch_data = extract_gfp_peaks(inst, tmin=tmin, tmax=tmax)
    assert isinstance(ch_data, ChData)


@pytest.mark.parametrize("inst", (raw, epochs))
def test_extract_gfp_invalid_arguments(inst):
    """Test errors raised when invalid arguments are provided."""
    with pytest.raises(TypeError, match="'inst' must be an instance of "):
        extract_gfp_peaks(101)
    with pytest.raises(TypeError, match="'min_peak_distance' must be an instance"):
        extract_gfp_peaks(inst, min_peak_distance=True)
    with pytest.raises(ValueError, match="Argument 'min_peak_distance' must be"):
        extract_gfp_peaks(inst, min_peak_distance=-2)
