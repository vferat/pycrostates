"""Test ModKMeans."""

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
from pycrostates.cluster import ModKMeans
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
ModK = ModKMeans(
    n_clusters=n_clusters, n_init=10, max_iter=100, tol=1e-4, random_state=1
)
ModK.fit(raw_eeg, n_jobs=1)


# pylint: disable=protected-access
def _check_fitted(ModK):
    """
    Checks that the ModK is fitted.
    """
    assert ModK.fitted
    assert ModK.n_clusters == n_clusters
    assert len(ModK._cluster_names) == n_clusters
    assert len(ModK._cluster_centers_) == n_clusters
    assert ModK._fitted_data is not None
    assert ModK._info is not None
    assert ModK.GEV_ is not None
    assert ModK._labels_ is not None


def _check_unfitted(ModK):
    """
    Checks that the ModK is not fitted.
    """
    assert not ModK.fitted
    assert ModK.n_clusters == n_clusters
    assert len(ModK._cluster_names) == n_clusters
    assert ModK._cluster_centers_ is None
    assert ModK._fitted_data is None
    assert ModK._info is None
    assert ModK.GEV_ is None
    assert ModK._labels_ is None


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


def test_ModKMeans():
    """Test K-Means default functionalities."""
    ModK1 = ModKMeans(
        n_clusters=n_clusters,
        n_init=10,
        max_iter=100,
        tol=1e-4,
        random_state=1,
    )

    # Test properties
    assert ModK1.n_init == 10
    assert ModK1.max_iter == 100
    assert ModK1.tol == 1e-4
    assert isinstance(ModK1.random_state, np.random.RandomState)
    _check_unfitted(ModK1)

    # Test default clusters names
    assert ModK1._cluster_names == ["0", "1", "2", "3"]

    # Test fit on RAW
    ModK1.fit(raw_eeg, n_jobs=1)
    _check_fitted(ModK1)
    assert ModK1._cluster_centers_.shape == (
        n_clusters,
        len(raw_eeg.info["ch_names"]) - len(raw_eeg.info["bads"]),
    )

    # Test reset
    ModK1.fitted = False
    _check_unfitted(ModK1)

    # Test fit on Epochs
    ModK1.fit(epochs_eeg, n_jobs=1)
    _check_fitted(ModK1)
    assert ModK1._cluster_centers_.shape == (
        n_clusters,
        len(epochs_eeg.info["ch_names"]) - len(epochs_eeg.info["bads"]),
    )

    # Test fit on ChData
    ModK1.fitted = False
    ModK1.fit(ch_data, n_jobs=1)
    _check_fitted(ModK1)
    assert ModK1._cluster_centers_.shape == (
        n_clusters,
        len(raw_eeg.info["ch_names"]) - len(raw_eeg.info["bads"]),
    )

    # Test copy
    ModK2 = ModK1.copy()
    _check_fitted(ModK2)
    assert np.isclose(ModK2._cluster_centers_, ModK1._cluster_centers_).all()
    assert np.isclose(ModK2.GEV_, ModK1.GEV_)
    assert np.isclose(ModK2._labels_, ModK1._labels_).all()
    ModK2.fitted = False
    _check_fitted(ModK1)
    _check_unfitted(ModK2)

    ModK3 = ModK1.copy(deep=False)
    _check_fitted(ModK3)
    assert np.isclose(ModK3._cluster_centers_, ModK1._cluster_centers_).all()
    assert np.isclose(ModK3.GEV_, ModK1.GEV_)
    assert np.isclose(ModK3._labels_, ModK1._labels_).all()
    ModK3.fitted = False
    _check_fitted(ModK1)
    _check_unfitted(ModK3)

    # Test representation
    expected = f"<ModKMeans | fitted on n = {n_clusters} cluster centers>"
    assert expected == ModK1.__repr__()
    assert "<ModKMeans | not fitted>" == ModK2.__repr__()

    # Test HTML representation
    html = ModK1._repr_html_()
    assert html is not None
    assert "not fitted" not in html
    html = ModK2._repr_html_()
    assert html is not None
    assert "not fitted" in html

    # Test plot
    f = ModK1.plot(block=False)
    assert isinstance(f, Figure)
    with pytest.raises(RuntimeError, match="must be fitted before"):
        ModK2.plot(block=False)
    plt.close("all")


def test_invert_polarity():
    """Test invert polarity method."""
    # list/tuple
    ModK_ = ModK.copy()
    cluster_centers_ = deepcopy(ModK_._cluster_centers_)
    ModK_.invert_polarity([True, False, True, False])
    assert np.isclose(
        ModK_._cluster_centers_[0, :], -cluster_centers_[0, :]
    ).all()
    assert np.isclose(
        ModK_._cluster_centers_[1, :], cluster_centers_[1, :]
    ).all()
    assert np.isclose(
        ModK_._cluster_centers_[2, :], -cluster_centers_[2, :]
    ).all()
    assert np.isclose(
        ModK_._cluster_centers_[3, :], cluster_centers_[3, :]
    ).all()

    # bool
    ModK_ = ModK.copy()
    cluster_centers_ = deepcopy(ModK_._cluster_centers_)
    ModK_.invert_polarity(True)
    assert np.isclose(
        ModK_._cluster_centers_[0, :], -cluster_centers_[0, :]
    ).all()
    assert np.isclose(
        ModK_._cluster_centers_[1, :], -cluster_centers_[1, :]
    ).all()
    assert np.isclose(
        ModK_._cluster_centers_[2, :], -cluster_centers_[2, :]
    ).all()
    assert np.isclose(
        ModK_._cluster_centers_[3, :], -cluster_centers_[3, :]
    ).all()

    # np.array
    ModK_ = ModK.copy()
    cluster_centers_ = deepcopy(ModK_._cluster_centers_)
    ModK_.invert_polarity(np.array([True, False, True, False]))
    assert np.isclose(
        ModK_._cluster_centers_[0, :], -cluster_centers_[0, :]
    ).all()
    assert np.isclose(
        ModK_._cluster_centers_[1, :], cluster_centers_[1, :]
    ).all()
    assert np.isclose(
        ModK_._cluster_centers_[2, :], -cluster_centers_[2, :]
    ).all()
    assert np.isclose(
        ModK_._cluster_centers_[3, :], cluster_centers_[3, :]
    ).all()

    # Test invalid arguments
    with pytest.raises(ValueError, match="not a 2D iterable"):
        ModK_.invert_polarity(np.zeros((2, 4)))
    with pytest.raises(
        ValueError, match=re.escape("list of bools of length 'n_clusters' (4)")
    ):
        ModK_.invert_polarity([True, False, True, False, True])
    with pytest.raises(TypeError, match="'invert' must be an instance of "):
        ModK_.invert_polarity(101)

    # Test unfitted
    ModK_.fitted = False
    _check_unfitted(ModK_)
    with pytest.raises(RuntimeError, match="must be fitted before"):
        ModK_.invert_polarity([True, False, True, False])


def test_rename(caplog):
    """Test renaming of clusters."""
    alphabet = ["A", "B", "C", "D"]

    # Test mapping
    ModK_ = ModK.copy()
    mapping = {old: alphabet[k] for k, old in enumerate(ModK._cluster_names)}
    for key, value in mapping.items():
        assert isinstance(key, str)
        assert isinstance(value, str)
        assert key != value
    ModK_.rename_clusters(mapping=mapping)
    assert ModK_._cluster_names == alphabet
    assert ModK_._cluster_names != ModK._cluster_names

    # Test new_names
    ModK_ = ModK.copy()
    ModK_.rename_clusters(new_names=alphabet)
    assert ModK_._cluster_names == alphabet
    assert ModK_._cluster_names != ModK._cluster_names

    # Test invalid arguments
    ModK_ = ModK.copy()
    with pytest.raises(TypeError, match="'mapping' must be an instance of "):
        ModK_.rename_clusters(mapping=101)
    with pytest.raises(ValueError, match="Invalid value for the 'old name'"):
        mapping = {
            old + "101": alphabet[k]
            for k, old in enumerate(ModK._cluster_names)
        }
        ModK_.rename_clusters(mapping=mapping)
    with pytest.raises(TypeError, match="'new name' must be an instance of "):
        mapping = {old: k for k, old in enumerate(ModK._cluster_names)}
        ModK_.rename_clusters(mapping=mapping)
    with pytest.raises(
        ValueError, match="Argument 'new_names' should contain"
    ):
        ModK_.rename_clusters(new_names=alphabet + ["E"])

    ModK_.rename_clusters()
    assert "Either 'mapping' or 'new_names' should not be" in caplog.text

    with pytest.raises(
        ValueError, match="Only one of 'mapping' or 'new_names'"
    ):
        mapping = {
            old: alphabet[k] for k, old in enumerate(ModK._cluster_names)
        }
        ModK_.rename_clusters(mapping=mapping, new_names=alphabet)

    # Test unfitted
    ModK_ = ModK.copy()
    ModK_.fitted = False
    _check_unfitted(ModK_)
    with pytest.raises(RuntimeError, match="must be fitted before"):
        mapping = {
            old: alphabet[k] for k, old in enumerate(ModK._cluster_names)
        }
        ModK_.rename_clusters(mapping=mapping)
    with pytest.raises(RuntimeError, match="must be fitted before"):
        ModK_.rename_clusters(new_names=alphabet)


def test_reorder(caplog):
    """Test reordering of clusters."""
    # Test mapping
    ModK_ = ModK.copy()
    ModK_.reorder_clusters(mapping={0: 1})
    assert np.isclose(
        ModK._cluster_centers_[0, :], ModK_._cluster_centers_[1, :]
    ).all()
    assert np.isclose(
        ModK._cluster_centers_[1, :], ModK_._cluster_centers_[0, :]
    ).all()
    assert ModK._cluster_names[0] == ModK_._cluster_names[1]
    assert ModK._cluster_names[0] == ModK_._cluster_names[1]

    # Test order
    ModK_ = ModK.copy()
    ModK_.reorder_clusters(order=[1, 0, 2, 3])
    assert np.isclose(
        ModK._cluster_centers_[0], ModK_._cluster_centers_[1]
    ).all()
    assert np.isclose(
        ModK._cluster_centers_[1], ModK_._cluster_centers_[0]
    ).all()
    assert ModK._cluster_names[0] == ModK_._cluster_names[1]
    assert ModK._cluster_names[0] == ModK_._cluster_names[1]

    ModK_ = ModK.copy()
    ModK_.reorder_clusters(order=np.array([1, 0, 2, 3]))
    assert np.isclose(
        ModK._cluster_centers_[0], ModK_._cluster_centers_[1]
    ).all()
    assert np.isclose(
        ModK._cluster_centers_[1], ModK_._cluster_centers_[0]
    ).all()
    assert ModK._cluster_names[0] == ModK_._cluster_names[1]
    assert ModK._cluster_names[0] == ModK_._cluster_names[1]

    # test ._labels_ reordering
    x = ModK_._labels_[:20]
    # x: before re-order:
    # x = [3, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    # y: expected re-ordered _labels
    y = [3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    assert np.all(x == y)

    # Test invalid arguments
    ModK_ = ModK.copy()
    with pytest.raises(TypeError, match="'mapping' must be an instance of "):
        ModK_.reorder_clusters(mapping=101)
    with pytest.raises(
        ValueError, match="Invalid value for the 'old position'"
    ):
        ModK_.reorder_clusters(mapping={4: 1})
    with pytest.raises(
        ValueError, match="Invalid value for the 'new position'"
    ):
        ModK_.reorder_clusters(mapping={0: 4})
    with pytest.raises(
        ValueError, match="Position in the new order can not be repeated."
    ):
        ModK_.reorder_clusters(mapping={0: 1, 2: 1})
    with pytest.raises(
        ValueError, match="A position can not be present in both"
    ):
        ModK_.reorder_clusters(mapping={0: 1, 1: 2})

    with pytest.raises(TypeError, match="'order' must be an instance of "):
        ModK_.reorder_clusters(order=101)
    with pytest.raises(ValueError, match="Invalid value for the 'order'"):
        ModK_.reorder_clusters(order=[4, 3, 1, 2])
    with pytest.raises(
        ValueError, match="Argument 'order' should contain 'n_clusters'"
    ):
        ModK_.reorder_clusters(order=[0, 3, 1, 2, 0])
    with pytest.raises(
        ValueError, match="Argument 'order' should be a 1D iterable"
    ):
        ModK_.reorder_clusters(order=np.array([[0, 1, 2, 3], [0, 1, 2, 3]]))

    ModK_.reorder_clusters()
    assert "Either 'mapping' or 'order' should not be 'None' " in caplog.text

    with pytest.raises(ValueError, match="Only one of 'mapping' or 'order'"):
        ModK_.reorder_clusters(mapping={0: 1}, order=[1, 0, 2, 3])

    # Test unfitted
    ModK_ = ModK.copy()
    ModK_.fitted = False
    _check_unfitted(ModK_)
    with pytest.raises(RuntimeError, match="must be fitted before"):
        ModK_.reorder_clusters(mapping={0: 1})
    with pytest.raises(RuntimeError, match="must be fitted before"):
        ModK_.reorder_clusters(order=[1, 0, 2, 3])


def test_properties(caplog):
    """Test properties."""
    caplog.set_level(logging.WARNING)

    # Unfitted
    ModK_ = ModKMeans(
        n_clusters=n_clusters,
        n_init=10,
        max_iter=100,
        tol=1e-4,
        random_state=1,
    )

    ModK_.cluster_centers_  # pylint: disable=pointless-statement
    assert "Clustering algorithm has not been fitted." in caplog.text
    caplog.clear()

    ModK_.info  # pylint: disable=pointless-statement
    assert "Clustering algorithm has not been fitted." in caplog.text
    caplog.clear()

    ModK_.fitted_data  # pylint: disable=pointless-statement
    assert "Clustering algorithm has not been fitted." in caplog.text
    caplog.clear()

    # Fitted
    ModK_ = ModK.copy()

    ModK_.cluster_centers_
    assert "Clustering algorithm has not been fitted." not in caplog.text
    caplog.clear()

    ModK_.info
    assert "Clustering algorithm has not been fitted." not in caplog.text
    caplog.clear()

    ModK_.fitted_data
    assert "Clustering algorithm has not been fitted." not in caplog.text
    caplog.clear()

    # Test fitted property
    ModK_ = ModKMeans(
        n_clusters=n_clusters,
        n_init=10,
        max_iter=100,
        tol=1e-4,
        random_state=1,
    )
    with pytest.raises(TypeError, match="'fitted' must be an instance of"):
        ModK_.fitted = "101"
    caplog.clear()
    ModK_.fitted = True
    log = "'fitted' can not be set to 'True' directly. Please use the .fit()"
    assert log in caplog.text
    caplog.clear()
    ModK_ = ModK.copy()
    ModK_.fitted = True
    log = "'fitted' can not be set to 'True' directly. The clustering"
    assert log in caplog.text


def test_invalid_arguments():
    """Test invalid arguments for init and for fit."""
    # n_clusters
    with pytest.raises(
        TypeError, match="'n_clusters' must be an instance of "
    ):
        ModK_ = ModKMeans(n_clusters="4")
    with pytest.raises(ValueError, match="The number of clusters must be a"):
        ModK_ = ModKMeans(n_clusters=0)
    with pytest.raises(ValueError, match="The number of clusters must be a"):
        ModK_ = ModKMeans(n_clusters=-101)

    # n_init
    with pytest.raises(TypeError, match="'n_init' must be an instance of "):
        ModK_ = ModKMeans(n_clusters=4, n_init="100")
    with pytest.raises(
        ValueError, match="The number of initialization must be a"
    ):
        ModK_ = ModKMeans(n_clusters=4, n_init=0)
    with pytest.raises(
        ValueError, match="The number of initialization must be a"
    ):
        ModK_ = ModKMeans(n_clusters=4, n_init=-101)

    # max_iter
    with pytest.raises(TypeError, match="'max_iter' must be an instance of "):
        ModK_ = ModKMeans(n_clusters=4, max_iter="100")
    with pytest.raises(
        ValueError, match="The number of max iteration must be a"
    ):
        ModK_ = ModKMeans(n_clusters=4, max_iter=0)
    with pytest.raises(
        ValueError, match="The number of max iteration must be a"
    ):
        ModK_ = ModKMeans(n_clusters=4, max_iter=-101)

    # tol
    with pytest.raises(TypeError, match="'tol' must be an instance of "):
        ModK_ = ModKMeans(n_clusters=4, tol="100")
    with pytest.raises(ValueError, match="The tolerance must be a"):
        ModK_ = ModKMeans(n_clusters=4, tol=0)
    with pytest.raises(ValueError, match="The tolerance must be a"):
        ModK_ = ModKMeans(n_clusters=4, tol=-101)

    # random-state
    with pytest.raises(ValueError, match="cannot be used to seed"):
        ModK_ = ModKMeans(n_clusters=4, random_state="101")

    ModK_ = ModKMeans(
        n_clusters=n_clusters,
        n_init=10,
        max_iter=100,
        tol=1e-4,
        random_state=1,
    )
    # inst
    with pytest.raises(TypeError, match="'inst' must be an instance of "):
        ModK_.fit(epochs_eeg.average())

    # tmin/tmax
    with pytest.raises(TypeError, match="'tmin' must be an instance of "):
        ModK_.fit(raw_eeg, tmin="101")
    with pytest.raises(TypeError, match="'tmax' must be an instance of "):
        ModK_.fit(raw_eeg, tmax="101")
    with pytest.raises(ValueError, match="Argument 'tmin' must be positive"):
        ModK_.fit(raw_eeg, tmin=-101, tmax=None)
    with pytest.raises(ValueError, match="Argument 'tmax' must be positive"):
        ModK_.fit(raw_eeg, tmin=None, tmax=-101)
    with pytest.raises(
        ValueError,
        match="Argument 'tmax' must be strictly larger than 'tmin'.",
    ):
        ModK_.fit(raw_eeg, tmin=5, tmax=1)
    with pytest.raises(
        ValueError,
        match="Argument 'tmin' must be shorter than the instance length.",
    ):
        ModK_.fit(raw_eeg, tmin=101, tmax=None)
    with pytest.raises(
        ValueError,
        match="Argument 'tmax' must be shorter than the instance length.",
    ):
        ModK_.fit(raw_eeg, tmin=None, tmax=101)

    # reject_by_annotation
    with pytest.raises(
        TypeError, match="'reject_by_annotation' must be an instance of "
    ):
        ModK_.fit(raw_eeg, reject_by_annotation=1)
    with pytest.raises(ValueError, match="only allows for"):
        ModK_.fit(raw_eeg, reject_by_annotation="101")


def test_fit_data_shapes():
    """Test different tmin/tmax, rejection with fit."""
    ModK_ = ModKMeans(
        n_clusters=n_clusters,
        n_init=10,
        max_iter=100,
        tol=1e-4,
        random_state=1,
    )

    # tmin
    ModK_.fitted = False
    _check_unfitted(ModK_)
    ModK_.fit(
        raw_eeg,
        n_jobs=1,
        picks="eeg",
        tmin=5,
        tmax=None,
        reject_by_annotation=False,
    )
    _check_fitted_data_raw(ModK_._fitted_data, raw_eeg, "eeg", 5, None, None)
    # save for later
    fitted_data_5_end = deepcopy(ModK_._fitted_data)

    ModK_.fitted = False
    _check_unfitted(ModK_)
    ModK_.fit(
        epochs_eeg,
        n_jobs=1,
        picks="eeg",
        tmin=0.2,
        tmax=None,
        reject_by_annotation=False,
    )
    _check_fitted_data_epochs(ModK_._fitted_data, epochs_eeg, "eeg", 0.2, None)

    # tmax
    ModK_.fitted = False
    _check_unfitted(ModK_)
    ModK_.fit(
        raw_eeg,
        n_jobs=1,
        picks="eeg",
        tmin=None,
        tmax=5,
        reject_by_annotation=False,
    )
    _check_fitted_data_raw(ModK_._fitted_data, raw_eeg, "eeg", None, 5, None)
    # save for later
    fitted_data_0_5 = deepcopy(ModK_._fitted_data)

    ModK_.fitted = False
    _check_unfitted(ModK_)
    ModK_.fit(
        epochs_eeg,
        n_jobs=1,
        picks="eeg",
        tmin=None,
        tmax=0.3,
        reject_by_annotation=False,
    )
    _check_fitted_data_epochs(ModK_._fitted_data, epochs_eeg, "eeg", None, 0.3)

    # tmin, tmax
    ModK_.fitted = False
    _check_unfitted(ModK_)
    ModK_.fit(
        raw_eeg,
        n_jobs=1,
        picks="eeg",
        tmin=2,
        tmax=8,
        reject_by_annotation=False,
    )
    _check_fitted_data_raw(ModK_._fitted_data, raw_eeg, "eeg", 2, 8, None)

    ModK_.fitted = False
    _check_unfitted(ModK_)
    ModK_.fit(
        epochs_eeg,
        n_jobs=1,
        picks="eeg",
        tmin=0.1,
        tmax=0.4,
        reject_by_annotation=False,
    )
    _check_fitted_data_epochs(ModK_._fitted_data, epochs_eeg, "eeg", 0.1, 0.4)

    # ---------------------
    # Reject by annotations
    # ---------------------
    bad_annot = Annotations([1], [2], "bad")
    raw_ = raw_eeg.copy()
    raw_.set_annotations(bad_annot)

    ModK_.fitted = False
    _check_unfitted(ModK_)

    ModK_no_reject = ModK_.copy()
    ModK_no_reject.fit(raw_, n_jobs=1, reject_by_annotation=False)
    ModK_reject_True = ModK_.copy()
    ModK_reject_True.fit(raw_, n_jobs=1, reject_by_annotation=True)
    ModK_reject_omit = ModK_.copy()
    ModK_reject_omit.fit(raw_, n_jobs=1, reject_by_annotation="omit")

    # Compare 'omit' and True
    assert np.isclose(
        ModK_reject_omit._fitted_data, ModK_reject_True._fitted_data
    ).all()
    assert np.isclose(ModK_reject_omit.GEV_, ModK_reject_True.GEV_)
    assert np.isclose(
        ModK_reject_omit._labels_, ModK_reject_True._labels_
    ).all()
    assert np.isclose(
        ModK_reject_omit._cluster_centers_, ModK_reject_True._cluster_centers_
    ).all()

    # Make sure there is a shape diff between True and False
    assert (
        ModK_reject_True._fitted_data.shape
        != ModK_no_reject._fitted_data.shape
    )

    # Check fitted data
    _check_fitted_data_raw(
        ModK_reject_True._fitted_data, raw_, "eeg", None, None, "omit"
    )
    _check_fitted_data_raw(
        ModK_no_reject._fitted_data, raw_, "eeg", None, None, None
    )

    # Check with reject with tmin/tmax
    ModK_rej_0_5 = ModK_.copy()
    ModK_rej_0_5.fit(raw_, n_jobs=1, tmin=0, tmax=5, reject_by_annotation=True)
    ModK_rej_5_end = ModK_.copy()
    ModK_rej_5_end.fit(
        raw_, n_jobs=1, tmin=5, tmax=None, reject_by_annotation=True
    )
    _check_fitted(ModK_rej_0_5)
    _check_fitted(ModK_rej_5_end)
    _check_fitted_data_raw(
        ModK_rej_0_5._fitted_data, raw_, "eeg", None, 5, "omit"
    )
    _check_fitted_data_raw(
        ModK_rej_5_end._fitted_data, raw_, "eeg", 5, None, "omit"
    )
    assert ModK_rej_0_5._fitted_data.shape != fitted_data_0_5.shape
    assert np.isclose(fitted_data_5_end, ModK_rej_5_end._fitted_data).all()


def test_refit():
    """Test that re-fit does not overwrite the current instance."""
    raw = raw_meg.copy().pick_types(meg=True, eeg=True, eog=True)
    ModK_ = ModKMeans(
        n_clusters=n_clusters,
        n_init=10,
        max_iter=100,
        tol=1e-4,
        random_state=1,
    )
    ModK_.fit(raw, picks="eeg")
    eeg_ch_names = ModK_.info["ch_names"]
    eeg_cluster_centers = ModK_.cluster_centers_
    ModK_.fitted = False  # unfit
    ModK_.fit(raw, picks="mag")
    mag_ch_names = ModK_.info["ch_names"]
    mag_cluster_centers = ModK_.cluster_centers_
    assert eeg_ch_names != mag_ch_names
    assert eeg_cluster_centers.shape != mag_cluster_centers.shape

    # invalid
    raw = raw_meg.copy().pick_types(meg=True, eeg=True, eog=True)
    ModK_ = ModKMeans(
        n_clusters=n_clusters,
        n_init=10,
        max_iter=100,
        tol=1e-4,
        random_state=1,
    )
    ModK_.fit(raw, picks="eeg")  # works
    eeg_ch_names = ModK_.info["ch_names"]
    eeg_cluster_centers = ModK_.cluster_centers_
    with pytest.raises(RuntimeError, match="must be unfitted"):
        ModK_.fit(raw, picks="mag")  # works
    assert eeg_ch_names == ModK_.info["ch_names"]
    assert np.allclose(eeg_cluster_centers, ModK_.cluster_centers_)


def test_predict_default(caplog):
    """Test predict method default behaviors."""
    # raw, no smoothing, no_edge
    segmentation = ModK.predict(raw_eeg, factor=0, reject_edges=False)
    assert isinstance(segmentation, RawSegmentation)
    assert "Segmenting data without smoothing" in caplog.text
    caplog.clear()

    # raw, no smoothing, with edge rejection
    segmentation = ModK.predict(raw_eeg, factor=0, reject_edges=True)
    assert isinstance(segmentation, RawSegmentation)
    assert segmentation._labels[0] == -1
    assert segmentation._labels[-1] == -1
    assert "Rejecting first and last segments." in caplog.text
    caplog.clear()

    # raw, with smoothing
    segmentation = ModK.predict(raw_eeg, factor=3, reject_edges=True)
    assert isinstance(segmentation, RawSegmentation)
    assert segmentation._labels[0] == -1
    assert segmentation._labels[-1] == -1
    assert "Segmenting data with factor 3" in caplog.text
    caplog.clear()

    # raw with min_segment_length
    segmentation = ModK.predict(
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
    segmentation = ModK.predict(epochs_eeg, factor=0, reject_edges=False)
    assert isinstance(segmentation, EpochsSegmentation)
    assert "Segmenting data without smoothing" in caplog.text
    caplog.clear()

    # epochs, no smoothing, with edge rejection
    segmentation = ModK.predict(epochs_eeg, factor=0, reject_edges=True)
    assert isinstance(segmentation, EpochsSegmentation)
    for epoch_labels in segmentation._labels:
        assert epoch_labels[0] == -1
        assert epoch_labels[-1] == -1
    assert "Rejecting first and last segments." in caplog.text
    caplog.clear()

    # epochs, with smoothing
    segmentation = ModK.predict(epochs_eeg, factor=3, reject_edges=True)
    assert isinstance(segmentation, EpochsSegmentation)
    for epoch_labels in segmentation._labels:
        assert epoch_labels[0] == -1
        assert epoch_labels[-1] == -1
    assert "Segmenting data with factor 3" in caplog.text
    caplog.clear()

    # epochs with min_segment_length
    segmentation = ModK.predict(
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
    segmentation_rej_True = ModK.predict(
        raw_, factor=0, reject_edges=True, reject_by_annotation=True
    )
    segmentation_rej_False = ModK.predict(
        raw_, factor=0, reject_edges=True, reject_by_annotation=False
    )
    segmentation_rej_None = ModK.predict(
        raw_, factor=0, reject_edges=True, reject_by_annotation=None
    )
    segmentation_no_annot = ModK.predict(
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
    segmentation1 = ModK.predict(
        raw_eeg, factor=3, reject_edges=False, half_window_size=3
    )
    segmentation2 = ModK.predict(
        raw_eeg, factor=3, reject_edges=False, half_window_size=60
    )
    segmentation3 = ModK.predict(
        raw_eeg, factor=0, reject_edges=False, half_window_size=3
    )
    assert not np.isclose(segmentation1._labels, segmentation2._labels).all()
    assert not np.isclose(segmentation1._labels, segmentation3._labels).all()
    assert not np.isclose(segmentation2._labels, segmentation3._labels).all()


def test_picks_fit_predict(caplog):
    """Test fitting and prediction with different picks."""
    raw = raw_meg.copy().pick_types(meg=True, eeg=True, eog=True)
    ModK_ = ModKMeans(
        n_clusters=n_clusters,
        n_init=10,
        max_iter=100,
        tol=1e-4,
        random_state=1,
    )

    # test invalid fit
    with pytest.raises(
        ValueError, match="Only one datatype can be selected for fitting"
    ):
        ModK_.fit(raw, picks=None)  # fails -> eeg + grad + mag
    with pytest.raises(
        ValueError, match="Only one datatype can be selected for fitting"
    ):
        ModK_.fit(raw, picks="meg")  # fails -> grad + mag
    with pytest.raises(
        ValueError, match="Only one datatype can be selected for fitting"
    ):
        ModK_.fit(raw, picks="data")  # fails -> eeg + grad + mag

    # test valid fit
    ModK_.fit(raw, picks="mag")  # works
    ModK_.fitted = False
    ModK_.fit(raw, picks="eeg")  # works
    ModK_.fitted = False

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
    ModK_.fit(raw, picks="eeg")
    assert ModK_.info["ch_names"] == ["Fp1", "CP1", "CP2"]
    assert "The channel Fp2 is set as bad and ignored" in caplog.text
    caplog.clear()

    # predict with the same channels in the instance used for prediction
    info = info_.copy()
    raw_predict = RawArray(data, info)
    caplog.clear()
    ModK_.predict(raw_predict, picks="eeg")  # -> warning for selected Fp2
    assert "Fp2 which was not used during fitting" in caplog.text
    caplog.clear()
    ModK_.predict(raw_predict, picks=["Fp1", "CP1", "CP2"])
    assert "Fp2 which was not used during fitting" not in caplog.text
    caplog.clear()
    raw_predict.info["bads"] = ["Fp2"]
    ModK_.predict(raw_predict, picks="eeg")
    assert "Fp2 which was not used during fitting" not in caplog.text

    # predict with a channel used for fitting that is now missing
    # fails, because ModK.info includes Fp1 which is bad in prediction instance
    raw_predict.info["bads"] = ["Fp1"]
    with pytest.raises(ValueError, match="Fp1 is required to predict"):
        ModK_.predict(raw_predict, picks="eeg")
    caplog.clear()

    # predict with a channel used for fitting that is now bad
    ModK_.predict(raw_predict, picks=["Fp1", "CP1", "CP2"])
    assert "Fp1 is set as bad in the instance but was selected" in caplog.text
    caplog.clear()

    # fails, because ModK_.info includes Fp1 which is missing from prediction
    # instance selection
    with pytest.raises(ValueError, match="Fp1 is required to predict"):
        ModK_.predict(raw_predict, picks=["CP2", "CP1"])

    # Try with one additional channel in the instance used for prediction.
    info_ = create_info(
        ["Fp1", "Fp2", "Fpz", "CP2", "CP1"], sfreq=1024, ch_types="eeg"
    )
    info_.set_montage("standard_1020")
    data = np.random.randn(5, 1024 * 10)
    raw_predict = RawArray(data, info_)

    # works, with warning because Fpz, Fp2 are missing from ModK_.info
    caplog.clear()
    ModK_.predict(raw_predict, picks="eeg")
    # handle non-deterministic sets
    msg1 = "Fp2, Fpz which were not used during fitting"
    msg2 = "Fpz, Fp2 which were not used during fitting"
    assert msg1 in caplog.text or msg2 in caplog.text
    caplog.clear()

    # fails, because ModK_.info includes Fp1 which is missing from prediction
    # instance selection
    with pytest.raises(ValueError, match="Fp1 is required to predict"):
        ModK_.predict(raw_predict, picks=["Fp2", "Fpz", "CP2", "CP1"])
    caplog.clear()

    # works, with warning because Fpz is missing from ModK_.info
    ModK_.predict(raw_predict, picks=["Fp1", "Fpz", "CP2", "CP1"])
    assert "Fpz which was not used during fitting" in caplog.text
    caplog.clear()

    # try with a missing channel from the prediction instance
    # fails, because Fp1 is used in ModK.info
    raw_predict.drop_channels(["Fp1"])
    with pytest.raises(
        ValueError, match="Fp1 was used during fitting but is missing"
    ):
        ModK_.predict(raw_predict, picks="eeg")

    # set a bad channel during fitting
    info = info_.copy()
    info["bads"] = ["Fp2"]
    raw = RawArray(data, info)

    ModK_.fitted = False
    caplog.clear()
    ModK_.fit(raw, picks=["Fp1", "Fp2", "CP2", "CP1"])
    assert ModK_.info["ch_names"] == ["Fp1", "Fp2", "CP2", "CP1"]
    assert "Fp2 is set as bad and will be used" in caplog.text
    caplog.clear()

    # predict with the same channels in the instance used for prediction
    info = info_.copy()
    raw_predict = RawArray(data, info)
    # works, with warning because a channel is bads in ModK_.info
    caplog.clear()
    ModK_.predict(raw_predict, picks="eeg")
    predict_warning = "fit contains bad channel Fp2 which will be used"
    assert predict_warning in caplog.text
    caplog.clear()

    # works, with warning because a channel is bads in ModK_.info
    raw_predict.info["bads"] = []
    ModK_.predict(raw_predict, picks=["Fp1", "Fp2", "CP2", "CP1"])
    assert predict_warning in caplog.text
    caplog.clear()

    # fails, because Fp2 is used in ModK_.info
    with pytest.raises(ValueError, match="Fp2 is required to predict"):
        ModK_.predict(raw_predict, picks=["Fp1", "CP2", "CP1"])

    # fails, because Fp2 is used in ModK_.info
    raw_predict.info["bads"] = ["Fp2"]
    with pytest.raises(ValueError, match="Fp2 is required to predict"):
        ModK_.predict(raw_predict, picks="eeg")

    # works, because same channels as ModK_.info
    caplog.clear()
    ModK_.predict(raw_predict, picks=["Fp1", "Fp2", "CP2", "CP1"])
    assert predict_warning in caplog.text
    assert "Fp2 is set as bad in the instance but was selected" in caplog.text
    caplog.clear()

    # fails, because ModK_.info includes Fp1 which is bad in prediction
    # instance
    raw_predict.info["bads"] = ["Fp1"]
    with pytest.raises(ValueError, match="Fp1 is required to predict because"):
        ModK_.predict(raw_predict, picks="eeg")

    # fails, because ModK_.info includes bad Fp2
    with pytest.raises(ValueError, match="Fp2 is required to predict"):
        ModK_.predict(raw_predict, picks=["Fp1", "CP2", "CP1"])

    # works, because same channels as ModK_.info (with warnings for Fp1, Fp2)
    caplog.clear()
    ModK_.predict(raw_predict, picks=["Fp1", "Fp2", "CP2", "CP1"])
    assert predict_warning in caplog.text
    assert "Fp1 is set as bad in the instance but was selected" in caplog.text
    caplog.clear()


def test_predict_invalid_arguments():
    """Test invalid arguments passed to predict."""
    with pytest.raises(TypeError, match="'inst' must be an instance of "):
        ModK.predict(epochs_eeg.average())
    with pytest.raises(TypeError, match="'factor' must be an instance of "):
        ModK.predict(raw_eeg, factor="0")
    with pytest.raises(
        TypeError, match="'reject_edges' must be an instance of "
    ):
        ModK.predict(raw_eeg, reject_edges=1)
    with pytest.raises(
        TypeError, match="'half_window_size' must be an instance of "
    ):
        ModK.predict(raw_eeg, half_window_size="1")
    with pytest.raises(TypeError, match="'tol' must be an instance of "):
        ModK.predict(raw_eeg, tol="0")
    with pytest.raises(
        TypeError, match="'min_segment_length' must be an instance of "
    ):
        ModK.predict(raw_eeg, min_segment_length="0")
    with pytest.raises(
        TypeError, match="'reject_by_annotation' must be an instance of "
    ):
        ModK.predict(raw_eeg, reject_by_annotation=1)
    with pytest.raises(ValueError, match="'reject_by_annotation' can be"):
        ModK.predict(raw_eeg, reject_by_annotation="101")


def test_n_jobs():
    """Test that n_jobs=2 works."""
    ModK_ = ModKMeans(
        n_clusters=n_clusters,
        n_init=10,
        max_iter=100,
        tol=1e-4,
        random_state=1,
    )
    ModK_.fit(raw_eeg, n_jobs=2)
    _check_fitted(ModK_)
    assert np.isclose(ModK_._cluster_centers_, ModK._cluster_centers_).all()
    assert np.isclose(ModK_.GEV_, ModK.GEV_)
    assert np.isclose(ModK_._labels_, ModK._labels_).all()


def test_fit_not_converged(caplog):
    """Test a fit that did not converge."""
    # 10/10 converged
    ModK_ = ModKMeans(
        n_clusters=n_clusters, n_init=10, max_iter=40, tol=1e-4, random_state=1
    )
    ModK_.fit(raw_eeg, n_jobs=1)
    _check_fitted(ModK_)
    assert "after 10/10 iterations converged." in caplog.text
    caplog.clear()

    # 6/10 converged
    ModK_ = ModKMeans(
        n_clusters=n_clusters,
        n_init=10,
        max_iter=20,
        tol=1e-10,
        random_state=1,
    )
    ModK_.fit(raw_eeg, n_jobs=1)
    _check_fitted(ModK_)
    assert "after 6/10 iterations converged." in caplog.text
    caplog.clear()

    # 0/10 converged
    ModK_ = ModKMeans(
        n_clusters=n_clusters,
        n_init=10,
        max_iter=10,
        tol=1e-20,
        random_state=1,
    )
    ModK_.fit(raw_eeg, n_jobs=1)
    _check_unfitted(ModK_)
    assert "All the K-means run failed to converge." in caplog.text
    caplog.clear()

    ModK_ = ModKMeans(
        n_clusters=n_clusters,
        n_init=10,
        max_iter=10,
        tol=1e-20,
        random_state=1,
    )
    ModK_.fit(raw_eeg, n_jobs=2)
    _check_unfitted(ModK_)
    assert "All the K-means run failed to converge." in caplog.text
    caplog.clear()


def test_randomseed():
    """Test that fit with same randomseed provide the same output."""
    ModK1 = ModKMeans(
        n_clusters=n_clusters, n_init=10, max_iter=40, tol=1e-4, random_state=1
    )
    ModK1.fit(raw_eeg, n_jobs=1)
    ModK2 = ModKMeans(
        n_clusters=n_clusters, n_init=10, max_iter=40, tol=1e-4, random_state=1
    )
    ModK2.fit(raw_eeg, n_jobs=1)
    ModK3 = ModKMeans(
        n_clusters=n_clusters, n_init=10, max_iter=40, tol=1e-4, random_state=2
    )
    ModK3.fit(raw_eeg, n_jobs=1)

    assert np.isclose(ModK1._cluster_centers_, ModK2._cluster_centers_).all()
    assert not np.isclose(
        ModK1._cluster_centers_, ModK3._cluster_centers_
    ).all()


def test_contains_mixin():
    """Test contains mixin class."""
    assert "eeg" in ModK
    assert ModK.compensation_grade is None
    assert ModK.get_channel_types() == ["eeg"] * ModK._info["nchan"]

    # test raise with non-fitted instance
    ModK_ = ModKMeans(
        n_clusters=n_clusters, n_init=10, max_iter=40, tol=1e-4, random_state=1
    )
    with pytest.raises(
        ValueError, match="Instance 'ModKMeans' attribute 'info' is None."
    ):
        "eeg" in ModK_
    with pytest.raises(
        ValueError, match="Instance 'ModKMeans' attribute 'info' is None."
    ):
        ModK_.get_channel_types()
    with pytest.raises(
        ValueError, match="Instance 'ModKMeans' attribute 'info' is None."
    ):
        ModK_.compensation_grade


def test_montage_mixin():
    """Test montage mixin class."""
    ModK_ = ModK.copy()
    montage = ModK.get_montage()
    assert isinstance(montage, DigMontage)
    assert montage.dig[-1]["r"][0] != 0
    montage.dig[-1]["r"][0] = 0
    ModK_.set_montage(montage)
    montage_ = ModK_.get_montage()
    assert montage_.dig[-1]["r"][0] == 0

    # test raise with non-fitted instance
    ModK_ = ModKMeans(
        n_clusters=n_clusters, n_init=10, max_iter=40, tol=1e-4, random_state=1
    )
    with pytest.raises(
        ValueError, match="Instance 'ModKMeans' attribute 'info' is None."
    ):
        ModK_.set_montage("standard_1020")

    with pytest.raises(
        ValueError, match="Instance 'ModKMeans' attribute 'info' is None."
    ):
        ModK_.get_montage()


def test_save(tmp_path, caplog):
    """Test .save() method."""
    # writing to .fif
    fname1 = tmp_path / "cluster.fif"
    ModK.save(fname1)

    # writing to .gz (compression)
    fname2 = tmp_path / "cluster.fif.gz"
    ModK.save(fname2)

    # re-load
    caplog.clear()
    ModK1 = read_cluster(fname1)
    assert __version__ in caplog.text
    caplog.clear()
    ModK2, version = _read_cluster(fname2)
    assert version == __version__
    assert __version__ not in caplog.text

    # compare
    assert ModK == ModK1
    assert ModK == ModK2
    assert ModK1 == ModK2  # sanity-check

    # test prediction
    segmentation = ModK.predict(raw_eeg, picks="eeg")
    segmentation1 = ModK1.predict(raw_eeg, picks="eeg")
    segmentation2 = ModK2.predict(raw_eeg, picks="eeg")

    assert np.allclose(segmentation._labels, segmentation1._labels)
    assert np.allclose(segmentation._labels, segmentation2._labels)
    assert np.allclose(segmentation1._labels, segmentation2._labels)


def test_comparison(caplog):
    """Test == and != methods."""
    ModK1 = ModK.copy()
    ModK2 = ModK.copy()
    assert ModK1 == ModK2

    # with different modkmeans variables
    ModK1.fitted = False
    assert ModK1 != ModK2
    ModK1 = ModK.copy()
    ModK1._n_init = 101
    assert ModK1 != ModK2
    ModK1 = ModK.copy()
    ModK1._max_iter = 101
    assert ModK1 != ModK2
    ModK1 = ModK.copy()
    ModK1._tol = 0.101
    assert ModK1 != ModK2
    ModK1 = ModK.copy()
    ModK1._GEV_ = 0.101
    assert ModK1 != ModK2

    # with different object
    assert ModK1 != 101

    # with different base variables
    ModK1 = ModK.copy()
    ModK2 = ModK.copy()
    assert ModK1 == ModK2
    ModK1 = ModK.copy()
    ModK1._n_clusters = 101
    assert ModK1 != ModK2
    ModK1 = ModK.copy()
    ModK1._info = ChInfo(
        ch_names=[str(k) for k in range(ModK1._cluster_centers_.shape[1])],
        ch_types=["eeg"] * ModK1._cluster_centers_.shape[1],
    )
    assert ModK1 != ModK2
    ModK1 = ModK.copy()
    ModK1._labels_ = ModK1._labels_[::-1]
    assert ModK1 != ModK2
    ModK1 = ModK.copy()
    ModK1._fitted_data = ModK1._fitted_data[:, ::-1]
    assert ModK1 != ModK2

    # different cluster names
    ModK1 = ModK.copy()
    ModK2 = ModK.copy()
    caplog.clear()
    assert ModK1 == ModK2
    assert "Cluster names differ between both clustering" not in caplog.text
    ModK1._cluster_names = ModK1._cluster_names[::-1]
    caplog.clear()
    assert ModK1 == ModK2
    assert "Cluster names differ between both clustering" in caplog.text
