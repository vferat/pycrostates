import re

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pycrostates.segmentation.transitions import (
    _check_labels_n_clusters,
    _compute_expected_transition_matrix,
    _compute_transition_matrix,
    compute_expected_transition_matrix,
    compute_transition_matrix,
)


@pytest.mark.parametrize(
    "labels, ignore_self, T",
    [
        # Raw
        (
            np.array([-1, 0, 1, 2, 3, 4, -1]),
            True,
            np.array(
                [
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0],
                ]
            ),
        ),
        (
            np.array([-1, -1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, -1, -1]),
            True,
            np.array(
                [
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0],
                ]
            ),
        ),
        (
            np.array([-1, -1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, -1, -1]),
            False,
            np.array(
                [
                    [0.5, 0.5, 0, 0, 0],
                    [0, 0.5, 0.5, 0, 0],
                    [0, 0, 0.5, 0.5, 0],
                    [0, 0, 0, 0.5, 0.5],
                    [0, 0, 0, 0, 1],
                ]
            ),
        ),
        # Epochs
        (
            np.array([[-1, 0], [1, 2], [3, 4], [-1, -1]]),
            True,
            np.array(
                [
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0],
                ]
            ),
        ),
    ],
)
def test_compute_transition_matrix(labels, ignore_self, T):
    n_clusters = (
        np.unique(labels).size - 1 if np.any(labels == -1) else np.unique(labels).size
    )
    t = _compute_transition_matrix(
        labels, n_clusters=n_clusters, ignore_self=ignore_self
    )
    assert isinstance(T, np.ndarray)
    assert t.shape == (n_clusters, n_clusters)
    assert_allclose(t, T)


def test_compute_expected_transition_matrix():
    # use bootstrap method to check results
    labels = np.random.randint(-1, 4, 500)
    n_clusters = 4
    Ts = []
    for _ in range(10000):
        labels_ = labels.copy()
        np.random.shuffle(labels_)
        T = _compute_transition_matrix(
            labels_, n_clusters, ignore_self=True, stat="probability"
        )
        Ts.append(T)
    boostrap_T = np.array(Ts).mean(axis=0)
    expected_T = _compute_expected_transition_matrix(
        labels, n_clusters, ignore_self=True, stat="probability"
    )
    assert_allclose(boostrap_T, expected_T, atol=1e-2)

    # case where 1 state is missing
    labels = np.random.randint(-1, 3, 500)
    n_clusters = 4
    Ts = []
    for _ in range(10000):
        labels_ = labels.copy()
        np.random.shuffle(labels_)
        T = _compute_transition_matrix(
            labels_, n_clusters, ignore_self=True, stat="probability"
        )
        Ts.append(T)
    boostrap_T = np.array(Ts).mean(axis=0)
    expected_T = _compute_expected_transition_matrix(
        labels, n_clusters, ignore_self=True, stat="probability"
    )
    assert_allclose(boostrap_T, expected_T, atol=1e-2)


def test_check_labels_n_clusters():
    """Test the static checker for the public entry-points."""
    # valids
    labels = np.random.randint(-1, 5, size=100)
    _check_labels_n_clusters(labels, 5)
    compute_transition_matrix(labels, 5)
    compute_expected_transition_matrix(labels, 5)
    labels = np.random.randint(0, 5, size=100)
    _check_labels_n_clusters(labels, 5)
    compute_transition_matrix(labels, 5)
    compute_expected_transition_matrix(labels, 5)

    # invalids
    with pytest.raises(ValueError, match="'-101' is invalid."):
        _check_labels_n_clusters(np.random.randint(-1, 5, size=100), -101)
    with pytest.raises(ValueError, match="Negative integers except -1 are invalid."):
        _check_labels_n_clusters(np.random.randint(-2, 5, size=100), 5)
    with pytest.raises(ValueError, match=re.escape("'[4]' is invalid.")):
        _check_labels_n_clusters(np.random.randint(1, 5, size=100), 4)
    with pytest.raises(ValueError, match="'float64' is invalid."):
        _check_labels_n_clusters(np.random.randint(0, 5, size=100).astype(float), 5)
    with pytest.raises(ValueError, match=re.escape("'[6 7]' are invalid")):
        _check_labels_n_clusters(np.random.randint(0, 8, size=100), 6)
