"""Test Metrics."""

import numpy as np
from mne.datasets import testing
from mne.io import read_raw_fif

from pycrostates.cluster import ModKMeans
from pycrostates.utils._clusters import optimize_order, _optimize_order


directory = testing.data_path() / "MEG" / "sample"
fname = directory / "sample_audvis_trunc_raw.fif"
raw = read_raw_fif(fname, preload=False)
raw.pick("eeg").crop(0, 10)
raw.load_data().apply_proj()
# Fit one for general purposes
n_clusters = 5
ModK_0 = ModKMeans(
    n_clusters=n_clusters, n_init=10, max_iter=100, tol=1e-4, random_state=0
)
ModK_0.fit(raw, n_jobs=1)

ModK_1 = ModKMeans(
    n_clusters=n_clusters, n_init=10, max_iter=100, tol=1e-4, random_state=1
)
ModK_1.fit(raw, n_jobs=1)


def test__optimize_order():
    n_states = 5
    n_electrodes = 3
    # Random template
    template = np.random.randint(-10, 10, (n_states, n_electrodes))
    # Shuffle template
    arr = np.arange(n_states)
    np.random.shuffle(arr)
    random_template = template[arr]
    # invert polarity
    polarities = np.random.choice([-1, 1], n_states)
    random_pol_template = polarities[:, np.newaxis] * random_template

    # No suffle
    current = template
    ignore_polarity = True
    order = _optimize_order(
        current, template, ignore_polarity=ignore_polarity
    )
    assert np.all(order == np.arange(n_states))

    # Shuffle
    current = random_template
    ignore_polarity = False
    order = _optimize_order(
        current, template, ignore_polarity=ignore_polarity
    )
    assert np.allclose(current[order], template)

    # Shuffle + ignore_polarity
    current = random_template
    ignore_polarity = True
    order = _optimize_order(
        current, template, ignore_polarity=ignore_polarity
    )
    assert np.allclose(current[order], template)

    # Shuffle + sign + ignore_polarity
    current = random_pol_template
    ignore_polarity = True
    order_ = _optimize_order(
        current, template, ignore_polarity=ignore_polarity
    )
    assert np.all(order == order_)

    # Shuffle + sign
    current = random_pol_template
    ignore_polarity = False
    order = _optimize_order(
        current, template, ignore_polarity=ignore_polarity
    )
    corr = np.corrcoef(template, current[order])[n_states:, :n_states]
    corr_order = np.corrcoef(template, current[order])[n_states:, :n_states]
    assert np.trace(corr) <= np.trace(corr_order)

def test_optimize_order():
    order = optimize_order(ModK_0, ModK_1)
    assert np.all(np.sort(np.unique(order)) == np.arange(len(order)))
    
    