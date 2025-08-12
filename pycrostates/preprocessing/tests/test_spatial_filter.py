import mne
import numpy as np
import pytest
from mne._fiff.pick import _picks_by_type
from mne.channels import find_ch_adjacency
from mne.datasets import testing

from pycrostates.io import ChData
from pycrostates.preprocessing import apply_spatial_filter

dir_ = testing.data_path() / "MEG" / "sample"
fname_raw_testing = dir_ / "sample_audvis_trunc_raw.fif"
raw_all = mne.io.read_raw_fif(fname_raw_testing, preload=False)
raw_all.info["bads"] = []
raw_all.crop(0, 10)
raw_all.load_data()
# raw
raw = raw_all.copy()
raw.pick("eeg")
# epochs
epochs = mne.make_fixed_length_epochs(raw, duration=2, preload=True)
# ch_data
ch_data = ChData(raw.get_data(), raw.info)


def test_spatial_filter_raw():
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


def test_spatial_filter_epochs():
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


def test_spatial_filter_ch_data():
    """Test apply_spatial_filter."""
    inst = ch_data
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


def test_spatial_filter_njobs():
    """Test apply_spatial_filter."""
    new_inst = apply_spatial_filter(raw, "eeg", n_jobs=2)
    assert isinstance(new_inst, type(raw))


def test_spatial_filter_eeg_and_meg():
    """Test apply_spatial_filter on raw with not only eeg channels."""
    new_inst = apply_spatial_filter(raw_all.copy(), "eeg")
    picks = dict(_picks_by_type(raw_all.info, exclude=[]))
    picks_eeg = picks["eeg"]
    picks_non_eeg = [
        idx for idx in np.arange(len(raw_all.ch_names)) if idx not in picks_eeg
    ]
    assert not np.all(new_inst._data[picks_eeg, :] == raw_all._data[picks_eeg, :])
    assert np.all(new_inst._data[picks_non_eeg, :] == raw_all._data[picks_non_eeg, :])


def test_spatial_filter_custom_adjacency():
    """Test apply_spatial_filter with custom adjacency."""
    adjacency_matrix, ch_names = find_ch_adjacency(raw_all.info, "eeg")
    apply_spatial_filter(raw_all.copy(), "eeg", adjacency=adjacency_matrix)
    with pytest.raises(ValueError, match="Adjacency must have exactly 2 dimensions"):
        apply_spatial_filter(raw_all.copy(), "eeg", adjacency=np.ones(len(ch_names)))
    with pytest.raises(ValueError, match="Adjacency must be of shape"):
        apply_spatial_filter(
            raw_all.copy(), "eeg", adjacency=adjacency_matrix[:-2, :-2]
        )
    with pytest.raises(
        ValueError, match="Values contained in adjacency can only be 0 or 1."
    ):
        adjacency_ = adjacency_matrix.copy()
        adjacency_[0, 0] = 2
        apply_spatial_filter(raw_all.copy(), "eeg", adjacency=adjacency_)
