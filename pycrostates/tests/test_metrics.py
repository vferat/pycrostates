import os.path as op

import mne
from mne.datasets import testing

from pycrostates.clustering import ModKMeans
from pycrostates.metrics import compute_metrics

data_path = testing.data_path(download=True)
fname_raw_testing = op.join(data_path, 'MEG', 'sample',
                            'sample_audvis_trunc_raw.fif')


def test_compute_fmetrics():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    raw = raw.filter(0, 40)
    raw = raw.crop(0, 10)
    n_clusters = 4
    ModK = ModKMeans(n_clusters=n_clusters)
    ModK.fit(raw)
    seg = ModK.predict(raw, reject_by_annotation=True)
    d = compute_metrics(seg, ModK.cluster_centers, raw)
    assert(isinstance(d,dict))