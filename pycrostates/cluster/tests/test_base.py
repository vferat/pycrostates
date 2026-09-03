"""Test staticmethod from base cluster class."""

import numpy as np
import pytest

from pycrostates.cluster._base import _BaseCluster


# pylint: disable=protected-access
def test_check_n_clusters():
    """Test the checker for n_clusters argument."""
    n_clusters = _BaseCluster._check_n_clusters(1)
    assert n_clusters == 1
    with pytest.raises(TypeError, match="'n_clusters' must be an instance of"):
        _BaseCluster._check_n_clusters(True)
    with pytest.raises(
        ValueError, match="number of clusters must be a positive integer"
    ):
        _BaseCluster._check_n_clusters(-101)


def test_reject_edge_segments():
    """Test method rejecting edge segments."""
    segmentation = np.array([1, 1, 2, 3, 2, 2, 3, 4, 4])
    segmentation = _BaseCluster._reject_edge_segments(segmentation)
    assert ([-1, -1, 2, 3, 2, 2, 3, -1, -1] == segmentation).all()

    segmentation = np.array([0, 1, 2, 3, 2, 2, 3, 0, 4])
    segmentation = _BaseCluster._reject_edge_segments(segmentation)
    assert ([-1, 1, 2, 3, 2, 2, 3, 0, -1] == segmentation).all()


def test_reject_short_segments():
    """Test method rejecting short segments."""
    segmentation = [0, 0, 1, 1, 1, 3, 3, 3, 1, 2, 2, 2, 2]
    data = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 3, 4, 0.5, 0.8, 1, 1, 1],
            [3, 3, 3, 3, 3, 3, 3, 6, 4, 5, 2, 2, 2],
        ]
    )
    segmentation = _BaseCluster._reject_short_segments(segmentation, data, 3)
    # solo 1 should turn to 2; initial 0 should not change
    assert [0, 0, 1, 1, 1, 3, 3, 3, 2, 2, 2, 2, 2] == segmentation

    # duet, same correlation
    segmentation = [0, 0, 1, 1, 1, 3, 3, 3, 1, 1, 2, 2, 2, 2]
    data = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 3, 2.5, 0.5, 0.5, 2.5, 1, 1, 1],
            [3, 3, 3, 3, 3, 3, 3, 6, 4, 4, 6, 2, 2, 2],
        ]
    )
    segmentation = _BaseCluster._reject_short_segments(segmentation, data, 3)
    assert [0, 0, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2] == segmentation

    # singleton, same correlation
    segmentation = [0, 0, 1, 1, 1, 3, 3, 3, 1, 2, 2, 2, 2]
    data = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 3, 2.5, 0.5, 2.5, 1, 1, 1],
            [3, 3, 3, 3, 3, 3, 3, 6, 4, 6, 2, 2, 2],
        ]
    )
    segmentation = _BaseCluster._reject_short_segments(segmentation, data, 3)
    assert [0, 0, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2] == segmentation


def test_smooth_segmentation_splits_on_unlabeled(monkeypatch):
    """Test that smoothing is applied independently on runs split by -1 labels."""
    # "11111222333-1-1111122233" -> sub-segments of length 11 and 9
    labels = np.array(
        [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, -1, -1, 1, 1, 1, 1, 2, 2, 2, 3, 3]
    )
    n_channels = 3
    data = np.zeros((n_channels, labels.size))
    states = np.zeros((4, n_channels))  # labels go up to 3

    calls = []

    def fake_smooth_segment(data_, states_, labels_, factor, tol, half_window_size):
        calls.append(labels_.copy())
        return np.full(labels_.shape, 99)

    monkeypatch.setattr(
        _BaseCluster, "_smooth_segment", staticmethod(fake_smooth_segment)
    )

    smoothed = _BaseCluster._smooth_segmentation(
        data, states, labels, factor=1, tol=1e-4, half_window_size=1
    )

    # only the 2 unlabeled-free sub-segments are smoothed independently
    assert len(calls) == 2
    assert calls[0].tolist() == labels[:11].tolist()
    assert calls[1].tolist() == labels[13:].tolist()

    expected = labels.copy()
    expected[:11] = 99
    expected[13:] = 99
    assert smoothed.tolist() == expected.tolist()
    # -1 samples are left untouched
    assert (smoothed[11:13] == -1).all()


def test_smooth_segmentation_skips_too_short_subsegments(monkeypatch):
    """Test that sub-segments shorter than the smoothing window are left untouched."""
    labels = np.array([1, 1, 1, -1, -1, 2, 2, 2])
    data = np.zeros((3, labels.size))
    states = np.zeros((3, 3))

    calls = []
    monkeypatch.setattr(
        _BaseCluster,
        "_smooth_segment",
        staticmethod(lambda *args, **kwargs: calls.append(1) or args[2]),
    )

    # half_window_size=2 requires sub-segments of at least 5 samples
    smoothed = _BaseCluster._smooth_segmentation(
        data, states, labels, factor=1, tol=1e-4, half_window_size=2
    )
    assert calls == []
    assert smoothed.tolist() == labels.tolist()

