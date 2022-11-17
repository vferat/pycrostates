import itertools

import mne
import numpy as np
import pytest
from mne import BaseEpochs
from mne.datasets import testing
from mne.io.pick import _picks_to_idx

from pycrostates.io import ChData
from pycrostates.preprocessing import apply_spatial_filter

dir_ = testing.data_path() / "MEG" / "sample"
fname_raw_testing = dir_ / "sample_audvis_trunc_raw.fif"
raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
raw.crop(0,10)
epochs = mne.make_fixed_length_epochs(raw, duration=2, preload=True)
ch_data = ChData(raw.get_data(), raw.info)

# pylint: disable=protected-access
@pytest.mark.parametrize("inst", (raw, epochs, ch_data))
def test_test_spatial_filter(inst):
    """Test apply_spatial_filter."""
    new_inst = apply_spatial_filter(inst, "eeg")
    assert isinstance(new_inst, type(inst))
