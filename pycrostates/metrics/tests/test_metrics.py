"""Test Metrics."""
from pathlib import Path

from mne.datasets import testing
from mne.io import read_raw_fif

from pycrostates.cluster import ModKMeans
from pycrostates.metrics import silhouette

directory = Path(testing.data_path()) / "MEG" / "sample"
fname = directory / "sample_audvis_trunc_raw.fif"
raw = read_raw_fif(fname, preload=True)
raw.pick("eeg").crop(0, 10).apply_proj()
# Fit one for general purposes
n_clusters = 5
ModK = ModKMeans(
    n_clusters=n_clusters, n_init=10, max_iter=100, tol=1e-4, random_state=1
)
ModK.fit(raw, n_jobs=1)


def test_silouhette():
    score = silhouette(ModK)
    assert isinstance(score, float)
