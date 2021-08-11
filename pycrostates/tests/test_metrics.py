import os.path as op

import numpy as np
import mne
from mne.datasets import testing

from pycrostates.clustering import ModKMeans
from pycrostates.metrics import silhouette, davies_bouldin, calinski_harabasz, dunn

data_path = testing.data_path()
fname_raw_testing = op.join(data_path, 'MEG', 'sample',
                            'sample_audvis_trunc_raw.fif')

raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
raw = raw.pick('eeg')
n_clusters = 4
ModK = ModKMeans(n_clusters=n_clusters, random_state=1)
ModK.fit(raw, n_jobs=1)

def test_silouhette():
    score = silhouette(ModK)
    assert isinstance(score, float)
    
def test_davies_bouldin():
    score = davies_bouldin(ModK)
    assert isinstance(score, float)
    
def test_calinski_harabasz():
    score = calinski_harabasz(ModK)
    assert isinstance(score, float)
    
def test_dunn():
    score = dunn(ModK)
    assert isinstance(score, float)