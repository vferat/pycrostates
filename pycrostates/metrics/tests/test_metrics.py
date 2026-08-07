"""Test Metrics."""

import numpy as np
from mne.datasets import testing
from mne.io import read_raw_fif
from numpy.testing import assert_allclose

from pycrostates.cluster import ModKMeans
from pycrostates.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    dunn_score,
    silhouette_score,
)

directory = testing.data_path() / "MEG" / "sample"
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


def test_metrics_invariant_to_global_polarity_flip():
    original = ModK
    flipped = ModK.copy(deep=True)
    flipped._fitted_data *= -1

    assert_allclose(silhouette_score(original), silhouette_score(flipped))
    assert_allclose(calinski_harabasz_score(original), calinski_harabasz_score(flipped))
    assert_allclose(dunn_score(original), dunn_score(flipped))
    assert_allclose(davies_bouldin_score(original), davies_bouldin_score(flipped))


def test_metrics_with_orthogonal_sample_are_stable():
    sample = 0
    modified = ModK.copy(deep=True)
    label = modified._labels_[sample]
    center = modified._cluster_centers_[label]

    rng = np.random.RandomState(0)
    candidate = rng.standard_normal(center.shape)
    orth = candidate - (candidate @ center) / (center @ center) * center
    orth /= np.linalg.norm(orth)
    modified._fitted_data[:, sample] = orth

    for scorer in (silhouette_score, calinski_harabasz_score, dunn_score, davies_bouldin_score):
        value = scorer(modified)
        assert np.isfinite(value)
