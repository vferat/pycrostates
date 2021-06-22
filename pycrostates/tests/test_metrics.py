import os.path as op

import mne
from mne.datasets import testing
from pycrostates.clustering import ModKMeans
from pycrostates.metrics import compute_metrics_data

data_path = testing.data_path()
fname_raw_testing = op.join(data_path, 'MEG', 'sample',
                            'sample_audvis_trunc_raw.fif')

