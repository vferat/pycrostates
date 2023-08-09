import numpy as np
import pytest
from mne import Epochs, make_fixed_length_events
from mne.datasets import testing
from mne.io import read_raw_fif

from pycrostates.cluster import ModKMeans
from pycrostates.utils._logs import logger, set_log_level

from ..entropy import _check_segmentation, _joint_entropy_history, entropy, excess_entropy_rate


set_log_level("INFO")
logger.propagate = True


dir_ = testing.data_path() / "MEG" / "sample"
fname_raw_testing = dir_ / "sample_audvis_trunc_raw.fif"
raw = read_raw_fif(fname_raw_testing, preload=False)
raw = raw.pick("eeg").crop(0, 10)
raw = raw.load_data().filter(1, 40).apply_proj()

events = make_fixed_length_events(raw, 1)
epochs = Epochs(raw, events, preload=True)

ModK_raw = ModKMeans(
    n_clusters=4, n_init=10, max_iter=100, tol=1e-4, random_state=1
)
ModK_epochs = ModKMeans(
    n_clusters=4, n_init=10, max_iter=100, tol=1e-4, random_state=1
)
ModK_raw.fit(raw, n_jobs=1)
ModK_epochs.fit(epochs, n_jobs=1)

raw_segmentation = ModK_raw.predict(raw)
epochs_segmentation = ModK_epochs.predict(epochs)

def test__check_segmentation():
    labels = np.random.randint(-1, 4, 100)
    r = _check_segmentation(labels)
    assert isinstance(r, np.ndarray)
    assert (r.ndim == 1)

    labels = np.random.randint(-1, 4, (100, 10))
    with pytest.raises(ValueError, match="must be a 1D array"
    ):
        _check_segmentation(labels)

    labels = np.random.uniform(0, 1, 100)
    with pytest.raises(ValueError, match="must be an array of integers"
    ):
        _check_segmentation(labels)

    r = _check_segmentation(raw_segmentation)
    assert isinstance(r, np.ndarray)
    assert (r.ndim == 1)

    r = _check_segmentation(epochs_segmentation)
    assert isinstance(r, np.ndarray)
    assert (r.ndim == 1)


def test__joint_entropy_history():
    labels = np.random.randint(-1, 4, 100)
    r = _joint_entropy_history(labels, 10, state_to_ignore=-1, log_base=2)
    assert isinstance(r, float)


def test_entropy():
    r = entropy(raw_segmentation, state_to_ignore=-1, ignore_self=False, log_base=2)
    assert isinstance(r, float)
    r_ = entropy(raw_segmentation._labels, state_to_ignore=-1, ignore_self=False, log_base=2)
    assert np.allclose(r, r_)

def test_excess_entropy_rate():
    a, b, residuals, lags, runs = excess_entropy_rate(raw_segmentation, history_length=10, state_to_ignore=-1, ignore_self=False, log_base=2)
    assert isinstance(a, float)
    assert isinstance(b, float)
    assert isinstance(residuals, float)
