import os.path as op

import numpy as np
import mne
from mne.datasets import testing

from pycrostates.clustering import ModKMeans
from pycrostates.segmentation import RawSegmentation, EpochsSegmentation, EvokedSegmentation
from pycrostates.metrics import silhouette

data_path = testing.data_path()
fname_raw_testing = op.join(data_path, 'MEG', 'sample',
                            'sample_audvis_trunc_raw.fif')

def test_silouhette():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    n_clusters = 4
    ModK = ModKMeans(n_clusters=n_clusters, random_state=1)
    ModK.fit(raw, n_jobs=1)
    s = silhouette(ModK)
    print(s)