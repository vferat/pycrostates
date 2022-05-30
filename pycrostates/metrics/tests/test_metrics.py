"""Test Metrics."""

from pathlib import Path

from mne.datasets import testing
from mne.io import read_raw_fif

from pycrostates.cluster import ModKMeans
from pycrostates.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    dunn_score,
    silhouette_score,
)

directory = Path(testing.data_path()) / "MEG" / "sample"
fname = directory / "sample_audvis_trunc_raw.fif"
raw = read_raw_fif(fname, preload=False)
raw.pick("eeg").crop(0, 10)
raw.load_data().apply_proj()
# Fit one for general purposes
n_clusters = 5
ModK = ModKMeans(
    n_clusters=n_clusters, n_init=10, max_iter=100, tol=1e-4, random_state=1
)
ModK.fit(raw, n_jobs=1)


def test_silouhette():
    score = silhouette_score(ModK)
    assert isinstance(score, float)


def test_calinski_harabasz():
    score = calinski_harabasz_score(ModK)
    assert isinstance(score, float)


def test_dunn():
    score = dunn_score(ModK)
    assert isinstance(score, float)


def test_davies_bouldin():
    score = davies_bouldin_score(ModK)
    assert isinstance(score, float)
