import numpy as np
import pytest

from pycrostates.segmentation.transitions import (
    _compute_expected_transition_matrix,
    _compute_transition_matrix,
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
        np.unique(labels).size - 1
        if np.any(labels == -1)
        else np.unique(labels).size
    )
    t = _compute_transition_matrix(
        labels, n_clusters=n_clusters, ignore_self=ignore_self
    )
    assert isinstance(T, np.ndarray)
    assert t.shape == (n_clusters, n_clusters)
    assert np.allclose(t, T)


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
    assert np.allclose(boostrap_T, expected_T, atol=1e-2)

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
    assert np.allclose(boostrap_T, expected_T, atol=1e-2)
