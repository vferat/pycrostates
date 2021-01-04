import pytest
import numpy as np
from pycrostates.clustering import mod_Kmeans


def test_mod_Kmeans():
    from mne.datasets import sample
    import mne
    data_path = sample.data_path()
    raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
    event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    raw = raw.pick('eeg')
    raw = raw.filter(0, 40)
    raw = raw.crop(0, 60)
    modK = mod_Kmeans()
    modK.fit(raw, n_jobs=2)
    assert modK.cluster_centers is not None