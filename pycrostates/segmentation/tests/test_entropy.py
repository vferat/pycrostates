import numpy as np
import pytest
from mne import make_fixed_length_epochs
from mne.datasets import testing
from mne.io import read_raw_fif

from pycrostates.cluster import ModKMeans
from pycrostates.utils._logs import logger, set_log_level

from ..entropy import (
    _auto_information,
    _check_labels,
    _check_lags,
    _check_segmentation,
    _entropy,
    _excess_entropy_rate,
    _joint_entropy,
    _joint_entropy_history,
    _partial_auto_information,
    auto_information_function,
    entropy,
    excess_entropy_rate,
    partial_auto_information_function,
)

set_log_level("INFO")
logger.propagate = True


dir_ = testing.data_path() / "MEG" / "sample"
fname_raw_testing = dir_ / "sample_audvis_trunc_raw.fif"
raw = read_raw_fif(fname_raw_testing, preload=False)
raw = raw.pick("eeg").crop(0, 10)
raw = raw.load_data().filter(1, 40).apply_proj()

epochs = make_fixed_length_epochs(raw, 1, preload=True)

ModK_raw = ModKMeans(n_clusters=4, n_init=10, max_iter=100, tol=1e-4, random_state=1)
ModK_epochs = ModKMeans(n_clusters=4, n_init=10, max_iter=100, tol=1e-4, random_state=1)
ModK_raw.fit(raw, n_jobs=1)
ModK_epochs.fit(epochs, n_jobs=1)

raw_segmentation = ModK_raw.predict(raw)
epochs_segmentation = ModK_epochs.predict(epochs)


def test__check_labels():
    labels = np.random.randint(-1, 4, 100)
    _check_labels(labels)

    labels = np.random.randint(-1, 4, (100, 10))
    with pytest.raises(ValueError, match="must be a 1D array"):
        _check_labels(labels)

    labels = np.random.uniform(0, 1, 100)
    with pytest.raises(ValueError, match="must be an array of integers"):
        _check_labels(labels)


def test__check_segmentation():
    r = _check_segmentation(raw_segmentation)
    assert isinstance(r, np.ndarray)
    assert r.ndim == 1

    r = _check_segmentation(epochs_segmentation)
    assert isinstance(r, np.ndarray)
    assert r.ndim == 1


def test_check_lags():
    _check_lags(np.arange(10))
    assert np.allclose(_check_lags(10), np.arange(10))
    assert np.allclose(_check_lags([1, 2, 3]), np.array([1, 2, 3]))
    assert np.allclose(_check_lags((1, 2, 3)), np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="If integer, lags must be >= 1."):
        _check_lags(-1)
    with pytest.raises(ValueError, match="Lags values must be positive."):
        _check_lags([1, 2, -3])
    with pytest.raises(ValueError, match="Lags values must be integers."):
        _check_lags([1, 2, 3.0])


def test__joint_entropy():
    x = np.random.randint(-1, 4, 100)
    y = np.random.randint(-1, 4, 100)
    r = _joint_entropy(x, y, state_to_ignore=-1, log_base=2)
    r_ = _joint_entropy(x, y, state_to_ignore=None, log_base=2)
    assert r != r_
    with pytest.raises(ValueError, match="Sequences of different lengths."):
        r = _joint_entropy(x, y[1:], state_to_ignore=None, log_base=2)


def test__joint_entropy_history():
    labels = np.random.randint(-1, 4, 100)
    r = _joint_entropy_history(labels, 10, state_to_ignore=-1, log_base=2)
    assert isinstance(r, float)
    r = _joint_entropy_history(labels, 10, state_to_ignore=None, log_base=2)
    assert isinstance(r, float)


def test__entropy():
    labels = np.random.randint(-1, 4, 100)
    r_ = _entropy(
        labels,
        state_to_ignore=-1,
        log_base=2,
    )
    assert isinstance(r_, float)

    r_ = _entropy(
        labels,
        state_to_ignore=None,
        log_base=2,
    )
    assert isinstance(r_, float)


def test_entropy():
    r = entropy(raw_segmentation, ignore_self=False, log_base=2)
    assert isinstance(r, float)

    r = entropy(epochs_segmentation, ignore_self=False, log_base=2)
    assert isinstance(r, float)

    r = entropy(raw_segmentation, ignore_self=True, log_base=2)
    assert isinstance(r, float)

    r = entropy(epochs_segmentation, ignore_self=False, log_base=10)
    assert isinstance(r, float)


def test__excess_entropy_rate():
    labels = np.random.randint(-1, 4, 100)
    a_, b_, residuals_, lags_, runs_ = _excess_entropy_rate(
        labels,
        history_length=10,
        state_to_ignore=-1,
        log_base=2,
    )
    assert isinstance(a_, float)
    assert isinstance(b_, float)
    assert isinstance(residuals_, float)

    # _excess_entropy_rate: state_to_ignore
    _excess_entropy_rate(
        labels,
        history_length=10,
        state_to_ignore=None,
        log_base=2,
    )


def test_excess_entropy_rate():
    # excess_entropy_rate: Raw
    excess_entropy_rate(
        raw_segmentation,
        history_length=10,
        ignore_self=False,
        log_base=2,
    )

    # excess_entropy_rate: Epochs
    excess_entropy_rate(
        epochs_segmentation,
        history_length=10,
        ignore_self=False,
        log_base=2,
    )

    # excess_entropy_rate: ignore_self
    excess_entropy_rate(
        raw_segmentation,
        history_length=10,
        ignore_self=True,
        log_base=2,
    )


def test__auto_information():
    labels = np.random.randint(-1, 4, 100)
    r = _auto_information(labels, 10, state_to_ignore=-1, log_base=2)
    assert isinstance(r, float)
    r = _auto_information(labels, 10, state_to_ignore=None, log_base=2)
    assert isinstance(r, float)


def test_auto_information_function():
    # Raw
    auto_information_function(
        raw_segmentation,
        lags=10,
        ignore_self=False,
        log_base=2,
    )
    # Epochs
    auto_information_function(
        epochs_segmentation,
        lags=10,
        ignore_self=False,
        log_base=2,
    )
    # ignore_self
    auto_information_function(
        raw_segmentation,
        lags=10,
        ignore_self=True,
        log_base=2,
    )
    # lags
    auto_information_function(
        raw_segmentation,
        lags=[1, 3, 10],
        ignore_self=True,
        log_base=2,
    )


def test__partial_auto_information():
    labels = np.random.randint(-1, 4, 100)
    r = _partial_auto_information(labels, 10, state_to_ignore=-1, log_base=2)
    assert isinstance(r, float)
    r = _partial_auto_information(labels, 10, state_to_ignore=None, log_base=2)
    assert isinstance(r, float)


def test_partial_auto_information_function():
    # Raw
    partial_auto_information_function(
        raw_segmentation,
        lags=10,
        ignore_self=False,
        log_base=2,
    )
    # Epochs
    partial_auto_information_function(
        epochs_segmentation,
        lags=10,
        ignore_self=False,
        log_base=2,
    )
    # ignore_self
    auto_information_function(
        raw_segmentation,
        lags=10,
        ignore_self=True,
        log_base=2,
    )
    # lags
    auto_information_function(
        raw_segmentation,
        lags=[1, 3, 10],
        ignore_self=True,
        log_base=2,
    )
