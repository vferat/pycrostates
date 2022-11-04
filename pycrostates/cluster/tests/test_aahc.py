"""Test AAHCluster."""

import logging
import re
from copy import deepcopy
from itertools import groupby

import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mne import Annotations, Epochs, create_info, make_fixed_length_events
from mne.channels import DigMontage
from mne.datasets import testing
from mne.io import RawArray, read_raw_fif
from mne.io.pick import _picks_to_idx

from pycrostates import __version__
from pycrostates.cluster import AAHCluster
from pycrostates.io import ChData, ChInfo, read_cluster
from pycrostates.io.fiff import _read_cluster
from pycrostates.segmentation import EpochsSegmentation, RawSegmentation
from pycrostates.utils._logs import logger, set_log_level

set_log_level("INFO")
logger.propagate = True

directory = testing.data_path() / "MEG" / "sample"
fname = directory / "sample_audvis_trunc_raw.fif"

# raw
raw_meg = read_raw_fif(fname, preload=False)
raw_meg.crop(0, 10)
raw_eeg = raw_meg.copy().pick("eeg").load_data().apply_proj()
raw_meg.pick_types(meg=True, eeg=True, exclude="bads")
raw_meg.load_data().apply_proj()
# epochs
events = make_fixed_length_events(raw_meg, duration=1)
epochs_meg = Epochs(
    raw_meg, events, tmin=0, tmax=0.5, baseline=None, preload=True
)
epochs_eeg = Epochs(
    raw_eeg, events, tmin=0, tmax=0.5, baseline=None, preload=True
)
# ch_data
ch_data = ChData(raw_eeg.get_data(), raw_eeg.info)
# Fit one for general purposes
n_clusters = 4

aah_cluster = AAHCluster(n_clusters=n_clusters, normalize_input=False)

aah_cluster.fit(ch_data)

# simulated data

# extract 3D positions from raw_eeg
pos = np.vstack([ch["loc"][:3] for ch in raw_eeg.info["chs"]])
# place 4 sources [3D origin, 3D orientation]
sources = np.array(
    [
        [0.000, 0.025, 0.060, 0.000, -0.050, 0.000],
        [0.000, 0.015, 0.080, 0.000, 0.025, 0.040],
        [0.000, -0.025, 0.050, 0.050, -0.040, 0.025],
        [0.000, -0.025, 0.050, -0.050, -0.040, 0.025],
    ],
    dtype=np.double,
)
sim_n_ms = sources.shape[0]
sim_n_frames = 250  # number of samples to generate
sim_n_chans = pos.shape[0]  # number of channels
# compute forward model
A = np.sum(
    (pos[None, ...] - sources[:, None, :3]) * sources[:, None, 3:], axis=2
)
A /= np.linalg.norm(A, axis=1, keepdims=True)
# simulate source actvities for 4 sources
# with positive and negative polarity
mapping = np.arange(sim_n_frames) % (sim_n_ms * 2)
s = (
    np.sign(mapping - sim_n_ms + 0.01)
    * np.eye(sim_n_ms)[:, mapping % sim_n_ms]
)
# apply forward model
X = A.T @ s
# add i.i.d. noise
sim_sigma = 0.05
X += sim_sigma * np.random.randn(*X.shape)
# generate the mne object
raw_sim = RawArray(X, raw_eeg.info, copy="info")
raw_sim.info["bads"] = []


def test_default_algorithm():
    obj = AAHCluster(n_clusters=sim_n_ms)
    assert obj._ignore_polarity is True  # pylint: disable=protected-access
    obj.fit(raw_sim)

    # extract cluster centers
    A_hat = obj.cluster_centers_

    # compute Euclidean distances (using the sign that minimizes the distance)
    sgn = np.sign(A @ A_hat.T)
    dists = np.linalg.norm(
        (A_hat[None, ...] - A[:, None] * sgn[..., None]), axis=2
    )
    # compute tolerance (2 times the expected noise level)
    tol = (
        2 * sim_sigma / np.sqrt(sim_n_frames / sim_n_ms) * np.sqrt(sim_n_chans)
    )
    # check if there is a cluster center whose distance
    # is within the tolerance
    assert (dists.min(axis=0) < tol).all()
    # ensure that all cluster centers were identified
    assert len(set(dists.argmin(axis=0))) == sim_n_ms


def test_ignore_polarity_false():
    obj = AAHCluster(n_clusters=sim_n_ms * 2)
    obj._ignore_polarity = False  # pylint: disable=protected-access
    obj.fit(raw_sim)

    # extract cluster centers
    A_hat = obj.cluster_centers_
    # create extended targets (pos. and neg. polarity)
    A_ = np.concatenate((A, -A), axis=0)
    # compute Euclidean distances
    dists = np.linalg.norm((A_hat[None, ...] - A_[:, None]), axis=2)
    # compute tolerance (2 times the expected noise level)
    tol = (
        2
        * sim_sigma
        / np.sqrt(sim_n_frames / sim_n_ms / 2)
        * np.sqrt(sim_n_chans)
    )
    # check if there is a cluster center whose distance
    # is within the tolerance
    assert (dists.min(axis=0) < tol).all()
    # ensure that all cluster centers were identified
    assert len(set(dists.argmin(axis=0))) == 2 * sim_n_ms


def test_normalize_input_true():
    obj = AAHCluster(
        n_clusters=sim_n_ms,
        # ignore_polarity=True,
        normalize_input=True,
    )
    assert obj._ignore_polarity is True  # pylint: disable=protected-access
    obj.fit(raw_sim)

    # extract cluster centers
    A_hat = obj.cluster_centers_

    # compute Euclidean distances (using the sign that minimizes the distance)
    sgn = np.sign(A @ A_hat.T)
    dists = np.linalg.norm(
        (A_hat[None, ...] - A[:, None] * sgn[..., None]), axis=2
    )
    # compute tolerance (2 times the expected noise level)
    tol = (
        2 * sim_sigma / np.sqrt(sim_n_frames / sim_n_ms) * np.sqrt(sim_n_chans)
    )
    # check if there is a cluster center whose distance
    # is within the tolerance
    assert (dists.min(axis=0) < tol).all()
    # ensure that all cluster centers were identified
    assert len(set(dists.argmin(axis=0))) == sim_n_ms


# pylint: disable=protected-access
def _check_fitted(aah_cluster):
    """Check that the aah_cluster is fitted."""
    assert aah_cluster.fitted
    assert aah_cluster.n_clusters == n_clusters
    assert len(aah_cluster._cluster_names) == n_clusters
    assert len(aah_cluster._cluster_centers_) == n_clusters
    assert aah_cluster._fitted_data is not None
    assert aah_cluster._info is not None
    assert aah_cluster.GEV_ is not None
    assert aah_cluster._labels_ is not None


def _check_unfitted(aah_cluster):
    """Check that the aah_cluster is not fitted."""
    assert not aah_cluster.fitted
    assert aah_cluster.n_clusters == n_clusters
    assert len(aah_cluster._cluster_names) == n_clusters
    assert aah_cluster._cluster_centers_ is None
    assert aah_cluster._fitted_data is None
    assert aah_cluster._info is None
    assert aah_cluster.GEV_ is None
    assert aah_cluster._labels_ is None


def _check_fitted_data_raw(
    fitted_data, raw, picks, tmin, tmax, reject_by_annotation
):
    """Check the fitted data array for a raw instance."""
    # Trust MNE .get_data() to correctly select data
    picks = _picks_to_idx(raw.info, picks)
    data = raw.get_data(
        picks=picks,
        tmin=tmin,
        tmax=tmax,
        reject_by_annotation=reject_by_annotation,
    )
    assert data.shape == fitted_data.shape


def _check_fitted_data_epochs(fitted_data, epochs, picks, tmin, tmax):
    """Check the fitted data array for an epoch instance."""
    picks = _picks_to_idx(epochs.info, picks)
    # Trust MNE .get_data() to correctly select data
    data = epochs.get_data(picks=picks, tmin=tmin, tmax=tmax)
    # check channels
    assert fitted_data.shape[0] == data.shape[1]
    # check samples
    assert fitted_data.shape[1] == int(data.shape[0] * data.shape[2])


def test_aahClusterMeans():
    """Test AAHC default functionalities."""
    aahCluster1 = AAHCluster(
        n_clusters=n_clusters,
        # ignore_polarity=True,
        normalize_input=False,
    )

    # Test properties
    # assert aahCluster1.ignore_polarity is True
    assert aahCluster1.normalize_input is False
    _check_unfitted(aahCluster1)

    # Test default clusters names
    assert aahCluster1._cluster_names == ["0", "1", "2", "3"]

    # Test fit on RAW
    aahCluster1.fit(raw_eeg)
    _check_fitted(aahCluster1)
    assert aahCluster1._cluster_centers_.shape == (
        n_clusters,
        len(raw_eeg.info["ch_names"]) - len(raw_eeg.info["bads"]),
    )

    # Test reset
    aahCluster1.fitted = False
    _check_unfitted(aahCluster1)

    # Test fit on Epochs
    aahCluster1.fit(epochs_eeg)
    _check_fitted(aahCluster1)
    assert aahCluster1._cluster_centers_.shape == (
        n_clusters,
        len(epochs_eeg.info["ch_names"]) - len(epochs_eeg.info["bads"]),
    )

    # Test fit on ChData
    aahCluster1.fitted = False
    aahCluster1.fit(ch_data)
    _check_fitted(aahCluster1)
    assert aahCluster1._cluster_centers_.shape == (
        n_clusters,
        len(raw_eeg.info["ch_names"]) - len(raw_eeg.info["bads"]),
    )

    # Test copy
    aahCluster2 = aahCluster1.copy()
    _check_fitted(aahCluster2)
    assert np.isclose(
        aahCluster2._cluster_centers_, aahCluster1._cluster_centers_
    ).all()
    assert np.isclose(aahCluster2.GEV_, aahCluster1.GEV_)
    assert np.isclose(aahCluster2._labels_, aahCluster1._labels_).all()
    aahCluster2.fitted = False
    _check_fitted(aahCluster1)
    _check_unfitted(aahCluster2)

    aahCluster3 = aahCluster1.copy(deep=False)
    _check_fitted(aahCluster3)
    assert np.isclose(
        aahCluster3._cluster_centers_, aahCluster1._cluster_centers_
    ).all()
    assert np.isclose(aahCluster3.GEV_, aahCluster1.GEV_)
    assert np.isclose(aahCluster3._labels_, aahCluster1._labels_).all()
    aahCluster3.fitted = False
    _check_fitted(aahCluster1)
    _check_unfitted(aahCluster3)

    # Test representation
    expected = f"<AAHCluster | fitted on n = {n_clusters} cluster centers>"
    assert expected == repr(aahCluster1)
    assert "<AAHCluster | not fitted>" == repr(aahCluster2)

    # Test HTML representation
    html = aahCluster1._repr_html_()
    assert html is not None
    assert "not fitted" not in html
    html = aahCluster2._repr_html_()
    assert html is not None
    assert "not fitted" in html

    # Test plot
    f = aahCluster1.plot(block=False)
    assert isinstance(f, Figure)
    with pytest.raises(RuntimeError, match="must be fitted before"):
        aahCluster2.plot(block=False)
    plt.close("all")


def test_invert_polarity():
    """Test invert polarity method."""
    # list/tuple
    aahCluster_ = aah_cluster.copy()
    cluster_centers_ = deepcopy(aahCluster_._cluster_centers_)
    aahCluster_.invert_polarity([True, False, True, False])
    assert np.isclose(
        aahCluster_._cluster_centers_[0, :], -cluster_centers_[0, :]
    ).all()
    assert np.isclose(
        aahCluster_._cluster_centers_[1, :], cluster_centers_[1, :]
    ).all()
    assert np.isclose(
        aahCluster_._cluster_centers_[2, :], -cluster_centers_[2, :]
    ).all()
    assert np.isclose(
        aahCluster_._cluster_centers_[3, :], cluster_centers_[3, :]
    ).all()

    # bool
    aahCluster_ = aah_cluster.copy()
    cluster_centers_ = deepcopy(aahCluster_._cluster_centers_)
    aahCluster_.invert_polarity(True)
    assert np.isclose(
        aahCluster_._cluster_centers_[0, :], -cluster_centers_[0, :]
    ).all()
    assert np.isclose(
        aahCluster_._cluster_centers_[1, :], -cluster_centers_[1, :]
    ).all()
    assert np.isclose(
        aahCluster_._cluster_centers_[2, :], -cluster_centers_[2, :]
    ).all()
    assert np.isclose(
        aahCluster_._cluster_centers_[3, :], -cluster_centers_[3, :]
    ).all()

    # np.array
    aahCluster_ = aah_cluster.copy()
    cluster_centers_ = deepcopy(aahCluster_._cluster_centers_)
    aahCluster_.invert_polarity(np.array([True, False, True, False]))
    assert np.isclose(
        aahCluster_._cluster_centers_[0, :], -cluster_centers_[0, :]
    ).all()
    assert np.isclose(
        aahCluster_._cluster_centers_[1, :], cluster_centers_[1, :]
    ).all()
    assert np.isclose(
        aahCluster_._cluster_centers_[2, :], -cluster_centers_[2, :]
    ).all()
    assert np.isclose(
        aahCluster_._cluster_centers_[3, :], cluster_centers_[3, :]
    ).all()

    # Test invalid arguments
    with pytest.raises(ValueError, match="not a 2D iterable"):
        aahCluster_.invert_polarity(np.zeros((2, 4)))
    with pytest.raises(
        ValueError, match=re.escape("list of bools of length 'n_clusters' (4)")
    ):
        aahCluster_.invert_polarity([True, False, True, False, True])
    with pytest.raises(TypeError, match="'invert' must be an instance of "):
        aahCluster_.invert_polarity(101)

    # Test unfitted
    aahCluster_.fitted = False
    _check_unfitted(aahCluster_)
    with pytest.raises(RuntimeError, match="must be fitted before"):
        aahCluster_.invert_polarity([True, False, True, False])


def test_rename(caplog):
    """Test renaming of clusters."""
    alphabet = ["A", "B", "C", "D"]

    # Test mapping
    aahCluster_ = aah_cluster.copy()
    mapping = {
        old: alphabet[k] for k, old in enumerate(aah_cluster._cluster_names)
    }
    for key, value in mapping.items():
        assert isinstance(key, str)
        assert isinstance(value, str)
        assert key != value
    aahCluster_.rename_clusters(mapping=mapping)
    assert aahCluster_._cluster_names == alphabet
    assert aahCluster_._cluster_names != aah_cluster._cluster_names

    # Test new_names
    aahCluster_ = aah_cluster.copy()
    aahCluster_.rename_clusters(new_names=alphabet)
    assert aahCluster_._cluster_names == alphabet
    assert aahCluster_._cluster_names != aah_cluster._cluster_names

    # Test invalid arguments
    aahCluster_ = aah_cluster.copy()
    with pytest.raises(TypeError, match="'mapping' must be an instance of "):
        aahCluster_.rename_clusters(mapping=101)
    with pytest.raises(ValueError, match="Invalid value for the 'old name'"):
        mapping = {
            old + "101": alphabet[k]
            for k, old in enumerate(aah_cluster._cluster_names)
        }
        aahCluster_.rename_clusters(mapping=mapping)
    with pytest.raises(TypeError, match="'new name' must be an instance of "):
        mapping = {old: k for k, old in enumerate(aah_cluster._cluster_names)}
        aahCluster_.rename_clusters(mapping=mapping)
    with pytest.raises(
        ValueError, match="Argument 'new_names' should contain"
    ):
        aahCluster_.rename_clusters(new_names=alphabet + ["E"])

    aahCluster_.rename_clusters()
    assert "Either 'mapping' or 'new_names' should not be" in caplog.text

    with pytest.raises(
        ValueError, match="Only one of 'mapping' or 'new_names'"
    ):
        mapping = {
            old: alphabet[k]
            for k, old in enumerate(aah_cluster._cluster_names)
        }
        aahCluster_.rename_clusters(mapping=mapping, new_names=alphabet)

    # Test unfitted
    aahCluster_ = aah_cluster.copy()
    aahCluster_.fitted = False
    _check_unfitted(aahCluster_)
    with pytest.raises(RuntimeError, match="must be fitted before"):
        mapping = {
            old: alphabet[k]
            for k, old in enumerate(aah_cluster._cluster_names)
        }
        aahCluster_.rename_clusters(mapping=mapping)
    with pytest.raises(RuntimeError, match="must be fitted before"):
        aahCluster_.rename_clusters(new_names=alphabet)


def test_reorder(caplog):
    """Test reordering of clusters."""
    # Test mapping
    aahCluster_ = aah_cluster.copy()
    aahCluster_.reorder_clusters(mapping={0: 1})
    assert np.isclose(
        aah_cluster._cluster_centers_[0, :],
        aahCluster_._cluster_centers_[1, :],
    ).all()
    assert np.isclose(
        aah_cluster._cluster_centers_[1, :],
        aahCluster_._cluster_centers_[0, :],
    ).all()
    assert aah_cluster._cluster_names[0] == aahCluster_._cluster_names[1]
    assert aah_cluster._cluster_names[0] == aahCluster_._cluster_names[1]

    # Test order
    aahCluster_ = aah_cluster.copy()
    aahCluster_.reorder_clusters(order=[1, 0, 2, 3])
    assert np.isclose(
        aah_cluster._cluster_centers_[0], aahCluster_._cluster_centers_[1]
    ).all()
    assert np.isclose(
        aah_cluster._cluster_centers_[1], aahCluster_._cluster_centers_[0]
    ).all()
    assert aah_cluster._cluster_names[0] == aahCluster_._cluster_names[1]
    assert aah_cluster._cluster_names[0] == aahCluster_._cluster_names[1]

    aahCluster_ = aah_cluster.copy()
    aahCluster_.reorder_clusters(order=np.array([1, 0, 2, 3]))
    assert np.isclose(
        aah_cluster._cluster_centers_[0], aahCluster_._cluster_centers_[1]
    ).all()
    assert np.isclose(
        aah_cluster._cluster_centers_[1], aahCluster_._cluster_centers_[0]
    ).all()
    assert aah_cluster._cluster_names[0] == aahCluster_._cluster_names[1]
    assert aah_cluster._cluster_names[0] == aahCluster_._cluster_names[1]

    # test ._labels_ reordering
    y = aah_cluster._labels_[:20]
    y[y == 0] = -1
    y[y == 1] = 0
    y[y == -1] = 1
    x = aahCluster_._labels_[:20]
    assert np.all(x == y)

    # Test invalid arguments
    aahCluster_ = aah_cluster.copy()
    with pytest.raises(TypeError, match="'mapping' must be an instance of "):
        aahCluster_.reorder_clusters(mapping=101)
    with pytest.raises(
        ValueError, match="Invalid value for the 'old position'"
    ):
        aahCluster_.reorder_clusters(mapping={4: 1})
    with pytest.raises(
        ValueError, match="Invalid value for the 'new position'"
    ):
        aahCluster_.reorder_clusters(mapping={0: 4})
    with pytest.raises(
        ValueError, match="Position in the new order can not be repeated."
    ):
        aahCluster_.reorder_clusters(mapping={0: 1, 2: 1})
    with pytest.raises(
        ValueError, match="A position can not be present in both"
    ):
        aahCluster_.reorder_clusters(mapping={0: 1, 1: 2})

    with pytest.raises(TypeError, match="'order' must be an instance of "):
        aahCluster_.reorder_clusters(order=101)
    with pytest.raises(ValueError, match="Invalid value for the 'order'"):
        aahCluster_.reorder_clusters(order=[4, 3, 1, 2])
    with pytest.raises(
        ValueError, match="Argument 'order' should contain 'n_clusters'"
    ):
        aahCluster_.reorder_clusters(order=[0, 3, 1, 2, 0])
    with pytest.raises(
        ValueError, match="Argument 'order' should be a 1D iterable"
    ):
        aahCluster_.reorder_clusters(
            order=np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
        )

    aahCluster_.reorder_clusters()
    assert "Either 'mapping' or 'order' should not be 'None' " in caplog.text

    with pytest.raises(ValueError, match="Only one of 'mapping' or 'order'"):
        aahCluster_.reorder_clusters(mapping={0: 1}, order=[1, 0, 2, 3])

    # Test unfitted
    aahCluster_ = aah_cluster.copy()
    aahCluster_.fitted = False
    _check_unfitted(aahCluster_)
    with pytest.raises(RuntimeError, match="must be fitted before"):
        aahCluster_.reorder_clusters(mapping={0: 1})
    with pytest.raises(RuntimeError, match="must be fitted before"):
        aahCluster_.reorder_clusters(order=[1, 0, 2, 3])


def test_properties(caplog):
    """Test properties."""
    caplog.set_level(logging.WARNING)

    # Unfitted
    aahCluster_ = AAHCluster(
        n_clusters=n_clusters,
        # ignore_polarity=True,
        normalize_input=False,
    )

    aahCluster_.cluster_centers_  # pylint: disable=pointless-statement
    assert "Clustering algorithm has not been fitted." in caplog.text
    caplog.clear()

    aahCluster_.info  # pylint: disable=pointless-statement
    assert "Clustering algorithm has not been fitted." in caplog.text
    caplog.clear()

    aahCluster_.fitted_data  # pylint: disable=pointless-statement
    assert "Clustering algorithm has not been fitted." in caplog.text
    caplog.clear()

    # Fitted
    aahCluster_ = aah_cluster.copy()

    assert aahCluster_.cluster_centers_ is not None
    assert "Clustering algorithm has not been fitted." not in caplog.text
    caplog.clear()

    assert aahCluster_.info is not None
    assert "Clustering algorithm has not been fitted." not in caplog.text
    caplog.clear()

    assert aahCluster_.fitted_data is not None
    assert "Clustering algorithm has not been fitted." not in caplog.text
    caplog.clear()

    # Test fitted property
    aahCluster_ = AAHCluster(
        n_clusters=n_clusters,
        # ignore_polarity=True,
        normalize_input=False,
    )
    with pytest.raises(TypeError, match="'fitted' must be an instance of"):
        aahCluster_.fitted = "101"
    caplog.clear()
    aahCluster_.fitted = True
    log = "'fitted' can not be set to 'True' directly. Please use the .fit()"
    assert log in caplog.text
    caplog.clear()
    aahCluster_ = aah_cluster.copy()
    aahCluster_.fitted = True
    log = "'fitted' can not be set to 'True' directly. The clustering"
    assert log in caplog.text


def test_invalid_arguments():
    """Test invalid arguments for init and for fit."""
    # n_clusters
    with pytest.raises(
        TypeError, match="'n_clusters' must be an instance of "
    ):
        aahCluster_ = AAHCluster(n_clusters="4")
    with pytest.raises(ValueError, match="The number of clusters must be a"):
        aahCluster_ = AAHCluster(n_clusters=0)
    with pytest.raises(ValueError, match="The number of clusters must be a"):
        aahCluster_ = AAHCluster(n_clusters=-101)

    # normalize_input
    with pytest.raises(
        TypeError, match="'normalize_input' must be an instance of bool"
    ):
        aahCluster_ = AAHCluster(n_clusters=n_clusters, normalize_input="asdf")
    with pytest.raises(
        TypeError, match="'normalize_input' must be an instance of bool"
    ):
        aahCluster_ = AAHCluster(n_clusters=n_clusters, normalize_input=None)

    aahCluster_ = AAHCluster(
        n_clusters=n_clusters,
        # ignore_polarity=True,
        normalize_input=False,
    )
    # inst
    with pytest.raises(TypeError, match="'inst' must be an instance of "):
        aahCluster_.fit(epochs_eeg.average())

    # tmin/tmax
    with pytest.raises(TypeError, match="'tmin' must be an instance of "):
        aahCluster_.fit(raw_eeg, tmin="101")
    with pytest.raises(TypeError, match="'tmax' must be an instance of "):
        aahCluster_.fit(raw_eeg, tmax="101")
    with pytest.raises(ValueError, match="Argument 'tmin' must be positive"):
        aahCluster_.fit(raw_eeg, tmin=-101, tmax=None)
    with pytest.raises(ValueError, match="Argument 'tmax' must be positive"):
        aahCluster_.fit(raw_eeg, tmin=None, tmax=-101)
    with pytest.raises(
        ValueError,
        match="Argument 'tmax' must be strictly larger than 'tmin'.",
    ):
        aahCluster_.fit(raw_eeg, tmin=5, tmax=1)
    with pytest.raises(
        ValueError,
        match="Argument 'tmin' must be shorter than the instance length.",
    ):
        aahCluster_.fit(raw_eeg, tmin=101, tmax=None)
    with pytest.raises(
        ValueError,
        match="Argument 'tmax' must be shorter than the instance length.",
    ):
        aahCluster_.fit(raw_eeg, tmin=None, tmax=101)

    # reject_by_annotation
    with pytest.raises(
        TypeError, match="'reject_by_annotation' must be an instance of "
    ):
        aahCluster_.fit(raw_eeg, reject_by_annotation=1)
    with pytest.raises(ValueError, match="only allows for"):
        aahCluster_.fit(raw_eeg, reject_by_annotation="101")


def test_fit_data_shapes():
    """Test different tmin/tmax, rejection with fit."""
    aahCluster_ = AAHCluster(
        n_clusters=n_clusters,
        # ignore_polarity=True,
        normalize_input=False,
    )

    # tmin
    aahCluster_.fitted = False
    _check_unfitted(aahCluster_)
    aahCluster_.fit(
        raw_eeg,
        picks="eeg",
        tmin=5,
        tmax=None,
        reject_by_annotation=False,
    )
    _check_fitted_data_raw(
        aahCluster_._fitted_data, raw_eeg, "eeg", 5, None, None
    )
    # save for later
    fitted_data_5_end = deepcopy(aahCluster_._fitted_data)

    aahCluster_.fitted = False
    _check_unfitted(aahCluster_)
    aahCluster_.fit(
        epochs_eeg,
        picks="eeg",
        tmin=0.2,
        tmax=None,
        reject_by_annotation=False,
    )
    _check_fitted_data_epochs(
        aahCluster_._fitted_data, epochs_eeg, "eeg", 0.2, None
    )

    # tmax
    aahCluster_.fitted = False
    _check_unfitted(aahCluster_)
    aahCluster_.fit(
        raw_eeg,
        picks="eeg",
        tmin=None,
        tmax=5,
        reject_by_annotation=False,
    )
    _check_fitted_data_raw(
        aahCluster_._fitted_data, raw_eeg, "eeg", None, 5, None
    )
    # save for later
    fitted_data_0_5 = deepcopy(aahCluster_._fitted_data)

    aahCluster_.fitted = False
    _check_unfitted(aahCluster_)
    aahCluster_.fit(
        epochs_eeg,
        picks="eeg",
        tmin=None,
        tmax=0.3,
        reject_by_annotation=False,
    )
    _check_fitted_data_epochs(
        aahCluster_._fitted_data, epochs_eeg, "eeg", None, 0.3
    )

    # tmin, tmax
    aahCluster_.fitted = False
    _check_unfitted(aahCluster_)
    aahCluster_.fit(
        raw_eeg,
        picks="eeg",
        tmin=2,
        tmax=8,
        reject_by_annotation=False,
    )
    _check_fitted_data_raw(
        aahCluster_._fitted_data, raw_eeg, "eeg", 2, 8, None
    )

    aahCluster_.fitted = False
    _check_unfitted(aahCluster_)
    aahCluster_.fit(
        epochs_eeg,
        picks="eeg",
        tmin=0.1,
        tmax=0.4,
        reject_by_annotation=False,
    )
    _check_fitted_data_epochs(
        aahCluster_._fitted_data, epochs_eeg, "eeg", 0.1, 0.4
    )

    # ---------------------
    # Reject by annotations
    # ---------------------
    bad_annot = Annotations([1], [2], "bad")
    raw_ = raw_eeg.copy()
    raw_.set_annotations(bad_annot)

    aahCluster_.fitted = False
    _check_unfitted(aahCluster_)

    aahCluster_no_reject = aahCluster_.copy()
    aahCluster_no_reject.fit(raw_, reject_by_annotation=False)
    aahCluster_reject_True = aahCluster_.copy()
    aahCluster_reject_True.fit(raw_, reject_by_annotation=True)
    aahCluster_reject_omit = aahCluster_.copy()
    aahCluster_reject_omit.fit(raw_, reject_by_annotation="omit")

    # Compare 'omit' and True
    assert np.isclose(
        aahCluster_reject_omit._fitted_data,
        aahCluster_reject_True._fitted_data,
    ).all()
    assert np.isclose(aahCluster_reject_omit.GEV_, aahCluster_reject_True.GEV_)
    assert np.isclose(
        aahCluster_reject_omit._labels_, aahCluster_reject_True._labels_
    ).all()
    # due to internal randomness, the sign can be flipped
    sgn = np.sign(
        np.sum(
            aahCluster_reject_True._cluster_centers_
            * aahCluster_reject_omit._cluster_centers_,
            axis=1,
        )
    )
    aahCluster_reject_True._cluster_centers_ *= sgn[:, None]
    assert np.isclose(
        aahCluster_reject_omit._cluster_centers_,
        aahCluster_reject_True._cluster_centers_,
    ).all()

    # Make sure there is a shape diff between True and False
    assert (
        aahCluster_reject_True._fitted_data.shape
        != aahCluster_no_reject._fitted_data.shape
    )

    # Check fitted data
    _check_fitted_data_raw(
        aahCluster_reject_True._fitted_data, raw_, "eeg", None, None, "omit"
    )
    _check_fitted_data_raw(
        aahCluster_no_reject._fitted_data, raw_, "eeg", None, None, None
    )

    # Check with reject with tmin/tmax
    aahCluster_rej_0_5 = aahCluster_.copy()
    aahCluster_rej_0_5.fit(raw_, tmin=0, tmax=5, reject_by_annotation=True)
    aahCluster_rej_5_end = aahCluster_.copy()
    aahCluster_rej_5_end.fit(
        raw_, tmin=5, tmax=None, reject_by_annotation=True
    )
    _check_fitted(aahCluster_rej_0_5)
    _check_fitted(aahCluster_rej_5_end)
    _check_fitted_data_raw(
        aahCluster_rej_0_5._fitted_data, raw_, "eeg", None, 5, "omit"
    )
    _check_fitted_data_raw(
        aahCluster_rej_5_end._fitted_data, raw_, "eeg", 5, None, "omit"
    )
    assert aahCluster_rej_0_5._fitted_data.shape != fitted_data_0_5.shape
    assert np.isclose(
        fitted_data_5_end, aahCluster_rej_5_end._fitted_data
    ).all()


def test_refit():
    """Test that re-fit does not overwrite the current instance."""
    raw = raw_meg.copy().pick_types(meg=True, eeg=True, eog=True)
    aahCluster_ = AAHCluster(
        n_clusters=n_clusters,
        # ignore_polarity=True,
        normalize_input=False,
    )
    aahCluster_.fit(raw, picks="eeg")
    eeg_ch_names = aahCluster_.info["ch_names"]
    eeg_cluster_centers = aahCluster_.cluster_centers_
    aahCluster_.fitted = False  # unfit
    aahCluster_.fit(raw, picks="mag")
    mag_ch_names = aahCluster_.info["ch_names"]
    mag_cluster_centers = aahCluster_.cluster_centers_
    assert eeg_ch_names != mag_ch_names
    assert eeg_cluster_centers.shape != mag_cluster_centers.shape

    # invalid
    raw = raw_meg.copy().pick_types(meg=True, eeg=True, eog=True)
    aahCluster_ = AAHCluster(
        n_clusters=n_clusters,
        # ignore_polarity=True,
        normalize_input=False,
    )
    aahCluster_.fit(raw, picks="eeg")  # works
    eeg_ch_names = aahCluster_.info["ch_names"]
    eeg_cluster_centers = aahCluster_.cluster_centers_
    with pytest.raises(RuntimeError, match="must be unfitted"):
        aahCluster_.fit(raw, picks="mag")  # works
    assert eeg_ch_names == aahCluster_.info["ch_names"]
    assert np.allclose(eeg_cluster_centers, aahCluster_.cluster_centers_)


# pylint: disable=too-many-statements
def test_predict_default(caplog):
    """Test predict method default behaviors."""
    # raw, no smoothing, no_edge
    segmentation = aah_cluster.predict(raw_eeg, factor=0, reject_edges=False)
    assert isinstance(segmentation, RawSegmentation)
    assert "Segmenting data without smoothing" in caplog.text
    caplog.clear()

    # raw, no smoothing, with edge rejection
    segmentation = aah_cluster.predict(raw_eeg, factor=0, reject_edges=True)
    assert isinstance(segmentation, RawSegmentation)
    assert segmentation._labels[0] == -1
    assert segmentation._labels[-1] == -1
    assert "Rejecting first and last segments." in caplog.text
    caplog.clear()

    # raw, with smoothing
    segmentation = aah_cluster.predict(raw_eeg, factor=3, reject_edges=True)
    assert isinstance(segmentation, RawSegmentation)
    assert segmentation._labels[0] == -1
    assert segmentation._labels[-1] == -1
    assert "Segmenting data with factor 3" in caplog.text
    caplog.clear()

    # raw with min_segment_length
    segmentation = aah_cluster.predict(
        raw_eeg, factor=0, reject_edges=False, min_segment_length=5
    )
    assert isinstance(segmentation, RawSegmentation)
    segment_lengths = [
        len(list(group)) for _, group in groupby(segmentation._labels)
    ]
    assert all(5 <= size for size in segment_lengths[1:-1])
    assert "Rejecting segments shorter than" in caplog.text
    caplog.clear()

    # epochs, no smoothing, no_edge
    segmentation = aah_cluster.predict(
        epochs_eeg, factor=0, reject_edges=False
    )
    assert isinstance(segmentation, EpochsSegmentation)
    assert "Segmenting data without smoothing" in caplog.text
    caplog.clear()

    # epochs, no smoothing, with edge rejection
    segmentation = aah_cluster.predict(epochs_eeg, factor=0, reject_edges=True)
    assert isinstance(segmentation, EpochsSegmentation)
    for epoch_labels in segmentation._labels:
        assert epoch_labels[0] == -1
        assert epoch_labels[-1] == -1
    assert "Rejecting first and last segments." in caplog.text
    caplog.clear()

    # epochs, with smoothing
    segmentation = aah_cluster.predict(epochs_eeg, factor=3, reject_edges=True)
    assert isinstance(segmentation, EpochsSegmentation)
    for epoch_labels in segmentation._labels:
        assert epoch_labels[0] == -1
        assert epoch_labels[-1] == -1
    assert "Segmenting data with factor 3" in caplog.text
    caplog.clear()

    # epochs with min_segment_length
    segmentation = aah_cluster.predict(
        epochs_eeg, factor=0, reject_edges=False, min_segment_length=5
    )
    assert isinstance(segmentation, EpochsSegmentation)
    for epoch_labels in segmentation._labels:
        segment_lengths = [
            len(list(group)) for _, group in groupby(epoch_labels)
        ]
        assert all(5 <= size for size in segment_lengths[1:-1])
    assert "Rejecting segments shorter than" in caplog.text
    caplog.clear()

    # raw with reject_by_annotation
    bad_annot = Annotations([1], [2], "bad")
    raw_ = raw_eeg.copy()
    raw_.set_annotations(bad_annot)
    segmentation_rej_True = aah_cluster.predict(
        raw_, factor=0, reject_edges=True, reject_by_annotation=True
    )
    segmentation_rej_False = aah_cluster.predict(
        raw_, factor=0, reject_edges=True, reject_by_annotation=False
    )
    segmentation_rej_None = aah_cluster.predict(
        raw_, factor=0, reject_edges=True, reject_by_annotation=None
    )
    segmentation_no_annot = aah_cluster.predict(
        raw_eeg, factor=0, reject_edges=True, reject_by_annotation="omit"
    )
    assert not np.isclose(
        segmentation_rej_True._labels, segmentation_rej_False._labels
    ).all()
    assert np.isclose(
        segmentation_no_annot._labels, segmentation_rej_False._labels
    ).all()
    assert np.isclose(
        segmentation_rej_None._labels, segmentation_rej_False._labels
    ).all()

    # test different half_window_size
    segmentation1 = aah_cluster.predict(
        raw_eeg, factor=3, reject_edges=False, half_window_size=3
    )
    segmentation2 = aah_cluster.predict(
        raw_eeg, factor=3, reject_edges=False, half_window_size=60
    )
    segmentation3 = aah_cluster.predict(
        raw_eeg, factor=0, reject_edges=False, half_window_size=3
    )
    assert not np.isclose(segmentation1._labels, segmentation2._labels).all()
    assert not np.isclose(segmentation1._labels, segmentation3._labels).all()
    assert not np.isclose(segmentation2._labels, segmentation3._labels).all()


# pylint: enable=too-many-statements


# pylint: disable=too-many-statements
def test_picks_fit_predict(caplog):
    """Test fitting and prediction with different picks."""
    raw = raw_meg.copy().pick_types(meg=True, eeg=True, eog=True)
    aahCluster_ = AAHCluster(
        n_clusters=n_clusters,
        # ignore_polarity=True,
        normalize_input=False,
    )

    # test invalid fit
    with pytest.raises(ValueError, match="Only one datatype can be selected"):
        aahCluster_.fit(raw, picks=None)  # fails -> eeg + grad + mag
    with pytest.raises(ValueError, match="Only one datatype can be selected"):
        aahCluster_.fit(raw, picks="meg")  # fails -> grad + mag
    with pytest.raises(ValueError, match="Only one datatype can be selected"):
        aahCluster_.fit(raw, picks="data")  # fails -> eeg + grad + mag

    # test valid fit
    aahCluster_.fit(raw, picks="mag")  # works
    aahCluster_.fitted = False
    aahCluster_.fit(raw, picks="eeg")  # works
    aahCluster_.fitted = False

    # create mock raw for fitting
    info_ = create_info(
        ["Fp1", "Fp2", "CP1", "CP2"], sfreq=1024, ch_types="eeg"
    )
    info_.set_montage("standard_1020")
    data = np.random.randn(4, 1024 * 10)

    # Ignore bad channel Fp2 during fitting
    info = info_.copy()
    info["bads"] = ["Fp2"]
    raw = RawArray(data, info)

    caplog.clear()
    aahCluster_.fit(raw, picks="eeg")
    assert aahCluster_.info["ch_names"] == ["Fp1", "CP1", "CP2"]
    assert "The channel Fp2 is set as bad and ignored" in caplog.text
    caplog.clear()

    # predict with the same channels in the instance used for prediction
    info = info_.copy()
    raw_predict = RawArray(data, info)
    caplog.clear()
    aahCluster_.predict(raw_predict, picks="eeg")
    # -> warning for selected Fp2
    assert "Fp2 which was not used during fitting" in caplog.text
    caplog.clear()
    aahCluster_.predict(raw_predict, picks=["Fp1", "CP1", "CP2"])
    assert "Fp2 which was not used during fitting" not in caplog.text
    caplog.clear()
    raw_predict.info["bads"] = ["Fp2"]
    aahCluster_.predict(raw_predict, picks="eeg")
    assert "Fp2 which was not used during fitting" not in caplog.text

    # predict with a channel used for fitting that is now missing
    # fails, because aah_cluster.info includes Fp1 which is bad
    # in prediction instance
    raw_predict.info["bads"] = ["Fp1"]
    with pytest.raises(ValueError, match="Fp1 is required to predict"):
        aahCluster_.predict(raw_predict, picks="eeg")
    caplog.clear()

    # predict with a channel used for fitting that is now bad
    aahCluster_.predict(raw_predict, picks=["Fp1", "CP1", "CP2"])
    assert "Fp1 is set as bad in the instance but was selected" in caplog.text
    caplog.clear()

    # fails, because aahCluster_.info includes Fp1 which is missing
    # from prediction instance selection
    with pytest.raises(ValueError, match="Fp1 is required to predict"):
        aahCluster_.predict(raw_predict, picks=["CP2", "CP1"])

    # Try with one additional channel in the instance used for prediction.
    info_ = create_info(
        ["Fp1", "Fp2", "Fpz", "CP2", "CP1"], sfreq=1024, ch_types="eeg"
    )
    info_.set_montage("standard_1020")
    data = np.random.randn(5, 1024 * 10)
    raw_predict = RawArray(data, info_)

    # works, with warning because Fpz, Fp2 are missing from aahCluster_.info
    caplog.clear()
    aahCluster_.predict(raw_predict, picks="eeg")
    # handle non-deterministic sets
    msg1 = "Fp2, Fpz which were not used during fitting"
    msg2 = "Fpz, Fp2 which were not used during fitting"
    assert msg1 in caplog.text or msg2 in caplog.text
    caplog.clear()

    # fails, because aahCluster_.info includes Fp1 which is missing from
    # prediction instance selection
    with pytest.raises(ValueError, match="Fp1 is required to predict"):
        aahCluster_.predict(raw_predict, picks=["Fp2", "Fpz", "CP2", "CP1"])
    caplog.clear()

    # works, with warning because Fpz is missing from aahCluster_.info
    aahCluster_.predict(raw_predict, picks=["Fp1", "Fpz", "CP2", "CP1"])
    assert "Fpz which was not used during fitting" in caplog.text
    caplog.clear()

    # try with a missing channel from the prediction instance
    # fails, because Fp1 is used in aah_cluster.info
    raw_predict.drop_channels(["Fp1"])
    with pytest.raises(
        ValueError, match="Fp1 was used during fitting but is missing"
    ):
        aahCluster_.predict(raw_predict, picks="eeg")

    # set a bad channel during fitting
    info = info_.copy()
    info["bads"] = ["Fp2"]
    raw = RawArray(data, info)

    aahCluster_.fitted = False
    caplog.clear()
    aahCluster_.fit(raw, picks=["Fp1", "Fp2", "CP2", "CP1"])
    assert aahCluster_.info["ch_names"] == ["Fp1", "Fp2", "CP2", "CP1"]
    assert "Fp2 is set as bad and will be used" in caplog.text
    caplog.clear()

    # predict with the same channels in the instance used for prediction
    info = info_.copy()
    raw_predict = RawArray(data, info)
    # works, with warning because a channel is bads in aahCluster_.info
    caplog.clear()
    aahCluster_.predict(raw_predict, picks="eeg")
    predict_warning = "fit contains bad channel Fp2 which will be used"
    assert predict_warning in caplog.text
    caplog.clear()

    # works, with warning because a channel is bads in aahCluster_.info
    raw_predict.info["bads"] = []
    aahCluster_.predict(raw_predict, picks=["Fp1", "Fp2", "CP2", "CP1"])
    assert predict_warning in caplog.text
    caplog.clear()

    # fails, because Fp2 is used in aahCluster_.info
    with pytest.raises(ValueError, match="Fp2 is required to predict"):
        aahCluster_.predict(raw_predict, picks=["Fp1", "CP2", "CP1"])

    # fails, because Fp2 is used in aahCluster_.info
    raw_predict.info["bads"] = ["Fp2"]
    with pytest.raises(ValueError, match="Fp2 is required to predict"):
        aahCluster_.predict(raw_predict, picks="eeg")

    # works, because same channels as aahCluster_.info
    caplog.clear()
    aahCluster_.predict(raw_predict, picks=["Fp1", "Fp2", "CP2", "CP1"])
    assert predict_warning in caplog.text
    assert "Fp2 is set as bad in the instance but was selected" in caplog.text
    caplog.clear()

    # fails, because aahCluster_.info includes Fp1 which is bad in prediction
    # instance
    raw_predict.info["bads"] = ["Fp1"]
    with pytest.raises(ValueError, match="Fp1 is required to predict because"):
        aahCluster_.predict(raw_predict, picks="eeg")

    # fails, because aahCluster_.info includes bad Fp2
    with pytest.raises(ValueError, match="Fp2 is required to predict"):
        aahCluster_.predict(raw_predict, picks=["Fp1", "CP2", "CP1"])

    # works, because same channels as aahCluster_.info
    # (with warnings for Fp1, Fp2)
    caplog.clear()
    aahCluster_.predict(raw_predict, picks=["Fp1", "Fp2", "CP2", "CP1"])
    assert predict_warning in caplog.text
    assert "Fp1 is set as bad in the instance but was selected" in caplog.text
    caplog.clear()


# pylint: enable=too-many-statements


def test_predict_invalid_arguments():
    """Test invalid arguments passed to predict."""
    with pytest.raises(TypeError, match="'inst' must be an instance of "):
        aah_cluster.predict(epochs_eeg.average())
    with pytest.raises(TypeError, match="'factor' must be an instance of "):
        aah_cluster.predict(raw_eeg, factor="0")
    with pytest.raises(
        TypeError, match="'reject_edges' must be an instance of "
    ):
        aah_cluster.predict(raw_eeg, reject_edges=1)
    with pytest.raises(
        TypeError, match="'half_window_size' must be an instance of "
    ):
        aah_cluster.predict(raw_eeg, half_window_size="1")
    with pytest.raises(
        TypeError, match="'min_segment_length' must be an instance of "
    ):
        aah_cluster.predict(raw_eeg, min_segment_length="0")
    with pytest.raises(
        TypeError, match="'reject_by_annotation' must be an instance of "
    ):
        aah_cluster.predict(raw_eeg, reject_by_annotation=1)
    with pytest.raises(ValueError, match="'reject_by_annotation' can be"):
        aah_cluster.predict(raw_eeg, reject_by_annotation="101")


def test_contains_mixin():
    """Test contains mixin class."""
    assert "eeg" in aah_cluster
    assert aah_cluster.compensation_grade is None
    assert (
        aah_cluster.get_channel_types() == ["eeg"] * aah_cluster._info["nchan"]
    )

    # test raise with non-fitted instance
    aahCluster_ = AAHCluster(
        n_clusters=n_clusters,
        # ignore_polarity=True,
        normalize_input=False,
    )
    with pytest.raises(
        ValueError, match="Instance 'AAHCluster' attribute 'info' is None."
    ):
        assert "eeg" in aahCluster_
    with pytest.raises(
        ValueError, match="Instance 'AAHCluster' attribute 'info' is None."
    ):
        aahCluster_.get_channel_types()
    with pytest.raises(
        ValueError, match="Instance 'AAHCluster' attribute 'info' is None."
    ):
        _ = aahCluster_.compensation_grade


def test_montage_mixin():
    """Test montage mixin class."""
    aahCluster_ = aah_cluster.copy()
    montage = aah_cluster.get_montage()
    assert isinstance(montage, DigMontage)
    assert montage.dig[-1]["r"][0] != 0
    montage.dig[-1]["r"][0] = 0
    aahCluster_.set_montage(montage)
    montage_ = aahCluster_.get_montage()
    assert montage_.dig[-1]["r"][0] == 0

    # test raise with non-fitted instance
    aahCluster_ = AAHCluster(
        n_clusters=n_clusters,
        # ignore_polarity=True,
        normalize_input=False,
    )
    with pytest.raises(
        ValueError, match="Instance 'AAHCluster' attribute 'info' is None."
    ):
        aahCluster_.set_montage("standard_1020")

    with pytest.raises(
        ValueError, match="Instance 'AAHCluster' attribute 'info' is None."
    ):
        aahCluster_.get_montage()


def test_save(tmp_path, caplog):
    """Test .save() method."""
    # writing to .fif
    fname1 = tmp_path / "cluster.fif"
    aah_cluster.save(fname1)

    # writing to .gz (compression)
    fname2 = tmp_path / "cluster.fif.gz"
    aah_cluster.save(fname2)

    # re-load
    caplog.clear()
    aahCluster1 = read_cluster(fname1)
    assert __version__ in caplog.text
    caplog.clear()
    aahCluster2, version = _read_cluster(fname2)
    assert version == __version__
    assert __version__ not in caplog.text

    # compare
    assert aah_cluster == aahCluster1
    assert aah_cluster == aahCluster2
    assert aahCluster1 == aahCluster2  # sanity-check

    # test prediction
    segmentation = aah_cluster.predict(raw_eeg, picks="eeg")
    segmentation1 = aahCluster1.predict(raw_eeg, picks="eeg")
    segmentation2 = aahCluster2.predict(raw_eeg, picks="eeg")

    assert np.allclose(segmentation._labels, segmentation1._labels)
    assert np.allclose(segmentation._labels, segmentation2._labels)
    assert np.allclose(segmentation1._labels, segmentation2._labels)


def test_comparison(caplog):
    """Test == and != methods."""
    aahCluster1 = aah_cluster.copy()
    aahCluster2 = aah_cluster.copy()
    assert aahCluster1 == aahCluster2

    # with different aahClustermeans variables
    aahCluster1.fitted = False
    assert aahCluster1 != aahCluster2
    aahCluster1 = aah_cluster.copy()
    aahCluster1._ignore_polarity = False
    assert aahCluster1 != aahCluster2
    aahCluster1 = aah_cluster.copy()
    aahCluster1._normalize_input = True
    assert aahCluster1 != aahCluster2
    aahCluster1 = aah_cluster.copy()
    aahCluster1._GEV_ = 0.101
    assert aahCluster1 != aahCluster2

    # with different object
    assert aahCluster1 != 101

    # with different base variables
    aahCluster1 = aah_cluster.copy()
    aahCluster2 = aah_cluster.copy()
    assert aahCluster1 == aahCluster2
    aahCluster1 = aah_cluster.copy()
    aahCluster1._n_clusters = 101
    assert aahCluster1 != aahCluster2
    aahCluster1 = aah_cluster.copy()
    aahCluster1._info = ChInfo(
        ch_names=[
            str(k) for k in range(aahCluster1._cluster_centers_.shape[1])
        ],
        ch_types=["eeg"] * aahCluster1._cluster_centers_.shape[1],
    )
    assert aahCluster1 != aahCluster2
    aahCluster1 = aah_cluster.copy()
    aahCluster1._labels_ = aahCluster1._labels_[::-1]
    assert aahCluster1 != aahCluster2
    aahCluster1 = aah_cluster.copy()
    aahCluster1._fitted_data = aahCluster1._fitted_data[:, ::-1]
    assert aahCluster1 != aahCluster2

    # different cluster names
    aahCluster1 = aah_cluster.copy()
    aahCluster2 = aah_cluster.copy()
    caplog.clear()
    assert aahCluster1 == aahCluster2
    assert "Cluster names differ between both clustering" not in caplog.text
    aahCluster1._cluster_names = aahCluster1._cluster_names[::-1]
    caplog.clear()
    assert aahCluster1 == aahCluster2
    assert "Cluster names differ between both clustering" in caplog.text
