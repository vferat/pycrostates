from pathlib import Path

import numpy as np
from mne.datasets import testing
from mne.io import read_raw_fif

from pycrostates.io import ChData

directory = Path(testing.data_path()) / "MEG" / "sample"
fname = directory / "sample_audvis_trunc_raw.fif"
raw = read_raw_fif(fname, preload=True)
data = raw.get_data()
info = raw.info


def test_create_Chdata():
    ch_data = ChData(data, info)
    assert np.allclose(ch_data.data, data)
    assert ch_data.info == info
