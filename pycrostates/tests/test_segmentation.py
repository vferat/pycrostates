import os.path as op

import mne
from mne.datasets import testing
from pycrostates.clustering import ModKMeans

data_path = testing.data_path()
fname_raw_testing = op.join(data_path, 'MEG', 'sample',
                            'sample_audvis_trunc_raw.fif')

raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
raw = raw.pick('eeg')
raw = raw.filter(0, 40)
raw = raw.crop(0, 10)
n_clusters = 4
ModK = ModKMeans(n_clusters=n_clusters)

def test_RawSegmentation_plot():
    ModK.fit(raw, n_jobs=1)
    segmentation = ModK.predict(raw)
    segmentation.plot()
    
    
def test_RawSegmentation_plot_cluster_centers():
    ModK.fit(raw, n_jobs=1)
    segmentation = ModK.predict(raw)
    segmentation.plot_cluster_centers()
    
    
def test_RawSegmentation_compute_metrics():
    ModK.fit(raw, n_jobs=1)
    segmentation = ModK.predict(raw)
    d = segmentation.compute_metrics(norm_gfp=True)
    assert isinstance(d,dict)