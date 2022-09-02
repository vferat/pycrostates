"""Test staticmethod from base cluster class."""

import numpy as np
import pytest
from mne import create_info
from mne.io.pick import _picks_to_idx

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


def test_check_picks_uniqueness():
    """Test method _check_picks_uniqueness."""
    # valid
    info = create_info(5, 1000, "eeg")
    picks = _picks_to_idx(info, "eeg")
    _BaseCluster._check_picks_uniqueness(info, picks)
    picks = _picks_to_idx(info, "all")
    _BaseCluster._check_picks_uniqueness(info, picks)

    info = create_info(5, 1000, ["eeg", "eeg", "eog", "ecg", "grad"])
    picks = _picks_to_idx(info, "eeg")
    _BaseCluster._check_picks_uniqueness(info, picks)
    picks = _picks_to_idx(info, "eog")
    _BaseCluster._check_picks_uniqueness(info, picks)
    picks = _picks_to_idx(info, "ecg")
    _BaseCluster._check_picks_uniqueness(info, picks)
    picks = _picks_to_idx(info, "meg")
    _BaseCluster._check_picks_uniqueness(info, picks)

    # invalid
    info = create_info(5, 1000, ["eeg", "eeg", "eog", "ecg", "grad"])
    picks = _picks_to_idx(info, "data")
    with pytest.raises(ValueError, match="Only one datatype can be selected"):
        _BaseCluster._check_picks_uniqueness(info, picks)
    picks = _picks_to_idx(info, "all")
    with pytest.raises(ValueError, match="Only one datatype can be selected"):
        _BaseCluster._check_picks_uniqueness(info, picks)
    picks = _picks_to_idx(info, [1, 2, 3])
    with pytest.raises(ValueError, match="Only one datatype can be selected"):
        _BaseCluster._check_picks_uniqueness(info, picks)


# TODO: Add tests for _smooth_segmentation and _segment?
