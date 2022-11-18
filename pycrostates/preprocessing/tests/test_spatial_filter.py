import mne
import numpy as np
from mne.datasets import testing

from pycrostates.preprocessing import apply_spatial_filter

dir_ = testing.data_path() / "MEG" / "sample"
fname_raw_testing = dir_ / "sample_audvis_trunc_raw.fif"
raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
raw.info["bads"] = []
# TODO: test with no EEG channels
raw.pick("eeg")
raw.crop(0, 10)
epochs = mne.make_fixed_length_epochs(raw, duration=2, preload=True)


def test_test_spatial_filter_raw():
    """Test apply_spatial_filter."""
    inst = raw
    inst_bad = inst.copy()
    inst_bad.info["bads"] = [inst.ch_names[0]]
    # exclude_bads=True
    # without bads
    new_inst = apply_spatial_filter(inst.copy(), "eeg", exclude_bads=True)
    assert new_inst.ch_names == inst.ch_names
    assert not np.all(new_inst._data[0] == inst._data[0])
    # with bads
    new_inst = apply_spatial_filter(inst_bad.copy(), "eeg", exclude_bads=True)
    assert new_inst.ch_names == inst.ch_names
    assert np.all(new_inst._data[0] == inst._data[0])
    assert not np.all(new_inst._data[1:] == inst._data[1:])
    # exclude_bads=False
    # without bads
    new_inst = apply_spatial_filter(inst.copy(), "eeg", exclude_bads=False)
    assert new_inst.ch_names == inst.ch_names
    assert not np.all(new_inst._data == inst._data)
    # with bads
    new_inst = apply_spatial_filter(inst_bad.copy(), "eeg", exclude_bads=True)
    assert new_inst.ch_names == inst.ch_names
    assert not np.all(new_inst._data == inst._data)


def test_test_spatial_filter_epochs():
    """Test apply_spatial_filter."""
    inst = epochs
    inst_bad = inst.copy()
    inst_bad.info["bads"] = [inst.ch_names[0]]
    # exclude_bads=True
    # without bads
    new_inst = apply_spatial_filter(inst.copy(), "eeg", exclude_bads=True)
    assert new_inst.ch_names == inst.ch_names
    assert not np.all(new_inst._data[:, 0, :] == inst._data[:, 0, :])
    # with bads
    new_inst = apply_spatial_filter(inst_bad.copy(), "eeg", exclude_bads=True)
    assert new_inst.ch_names == inst.ch_names
    assert np.all(new_inst._data[:, 0, :] == inst._data[:, 0, :])
    assert not np.all(new_inst._data[:, 1:, :] == inst._data[:, 1:, :])
    # exclude_bads=False
    # without bads
    new_inst = apply_spatial_filter(inst.copy(), "eeg", exclude_bads=False)
    assert new_inst.ch_names == inst.ch_names
    assert not np.all(new_inst._data == inst._data)
    # with bads
    new_inst = apply_spatial_filter(inst_bad.copy(), "eeg", exclude_bads=True)
    assert new_inst.ch_names == inst.ch_names
    assert not np.all(new_inst._data == inst._data)


def test_spatial_filter_njobs():
    """Test apply_spatial_filter."""
    new_inst = apply_spatial_filter(raw, "eeg", n_jobs=2)
    assert isinstance(new_inst, type(raw))
