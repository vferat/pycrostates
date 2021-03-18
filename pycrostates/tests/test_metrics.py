import os.path as op

import mne
from mne.datasets import testing
from pycrostates.clustering import ModKMeans
from pycrostates.metrics import compute_metrics

data_path = testing.data_path()
fname_raw_testing = op.join(data_path, 'MEG', 'sample',
                            'sample_audvis_trunc_raw.fif')


def test_compute_metrics():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    raw = raw.filter(0, 40)
    raw = raw.crop(0, 10)
    n_clusters = 4
    ModK = ModKMeans(n_clusters=n_clusters)
    ModK.fit(raw)
    d = compute_metrics(raw,
                    modK = ModK,
                    norm_gfp=True,
                    reject_by_annotation=True,
                    half_window_size=3, factor=0,
                    crit=10e-6,
                    n_jobs=1,
                    verbose=None)
    assert(isinstance(d,list))
    assert(isinstance(d[0],dict))

def test_compute_metrics_list():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    raw = raw.filter(0, 40)
    raw = raw.crop(0, 10)
    n_clusters = 4
    ModK = ModKMeans(n_clusters=n_clusters)
    ModK.fit(raw)
    d = compute_metrics(inst=[raw,raw],
                    modK = ModK,
                    norm_gfp=True,
                    reject_by_annotation=True,
                    half_window_size=3, factor=0,
                    crit=10e-6,
                    n_jobs=1,
                    verbose=None)
    assert(isinstance(d,list))
    assert(len(d) == 2)
    
def test_compute_metrics_list_parallel():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    raw = raw.filter(0, 40)
    raw = raw.crop(0, 10)
    n_clusters = 4
    ModK = ModKMeans(n_clusters=n_clusters)
    ModK.fit(raw)
    d = compute_metrics(inst=[raw,raw],
                    modK = ModK,
                    norm_gfp=True,
                    reject_by_annotation=True,
                    half_window_size=3, factor=0,
                    crit=10e-6,
                    n_jobs=2,
                    verbose=None)
    assert(isinstance(d,list))
    assert(len(d) == 2)