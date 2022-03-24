from copy import deepcopy
from itertools import groupby
import logging
import re
from pathlib import Path

from matplotlib import pyplot as plt
import mne
from mne.datasets import testing
from mne.io.pick import _picks_to_idx
import numpy as np
import pytest

from pycrostates.cluster import ModKMeans
from pycrostates.segmentation import RawSegmentation, EpochsSegmentation
from pycrostates.utils._logs import logger, set_log_level


set_log_level('INFO')
logger.propagate = True


directory = Path(testing.data_path()) / 'MEG' / 'sample'
fname = directory / 'sample_audvis_trunc_raw.fif'
raw = mne.io.read_raw_fif(fname, preload=True)
raw.pick('eeg').crop(0, 10).apply_proj()
events = mne.make_fixed_length_events(raw, duration=1)
epochs = mne.Epochs(raw, events, tmin=0, tmax=0.5, baseline=None, preload=True)
n_clusters = 4

# Fit one for general purposes
ModK = ModKMeans(n_clusters=n_clusters, n_init=10, max_iter=100, tol=1e-4,
                 random_state=1)
ModK.fit(raw, n_jobs=1)


def _check_fitted(ModK):
    """
    Checks that the ModK is fitted.
    """
    assert ModK.fitted
    assert ModK.n_clusters == n_clusters
    assert len(ModK.clusters_names) == n_clusters
    assert len(ModK.cluster_centers_) == n_clusters
    assert ModK.fitted_data is not None
    assert ModK.picks is not None
    assert ModK.info is not None
    assert ModK.GEV_ is not None
    assert ModK.labels_ is not None


def _check_unfitted(ModK):
    """
    Checks that the ModK is not fitted.
    """
    assert not ModK.fitted
    assert ModK.n_clusters == n_clusters
    assert len(ModK.clusters_names) == n_clusters
    assert ModK.cluster_centers_ is None
    assert ModK.fitted_data is None
    assert ModK.picks is None
    assert ModK.info is None
    assert ModK.GEV_ is None
    assert ModK.labels_ is None


def _check_fitted_data_raw(fitted_data, raw, picks, tmin, tmax,
                           reject_by_annotation):
    """Check the fitted data array for a raw instance."""
    # Trust MNE .get_data() to correctly select data
    picks = _picks_to_idx(raw.info, picks)
    data = raw.get_data(picks=picks, tmin=tmin, tmax=tmax,
                        reject_by_annotation=reject_by_annotation)
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
    ModK1 = ModKMeans(n_clusters=n_clusters, n_init=10, max_iter=100, tol=1e-4,
                      random_state=1)

    # Test properties
    assert ModK1.n_init == 10
    assert ModK1.max_iter == 100
    assert ModK1.tol == 1e-4
    assert isinstance(ModK1.random_state, np.random.RandomState)
    _check_unfitted(ModK1)

    # Test default clusters names
    assert ModK1.clusters_names == ['0', '1', '2', '3']

    # Test fit on RAW
    ModK1.fit(raw, n_jobs=1)
    _check_fitted(ModK1)
    assert ModK1.cluster_centers_.shape == \
        (n_clusters, len(raw.info['ch_names']) - len(raw.info['bads']))

    # Test reset
    ModK1.fitted = False
    _check_unfitted(ModK1)

    # Test fit on Epochs
    ModK1.fit(epochs, n_jobs=1)
    _check_fitted(ModK1)
    assert ModK1.cluster_centers_.shape == \
        (n_clusters, len(epochs.info['ch_names']) - len(epochs.info['bads']))

    # Test copy
    ModK2 = ModK1.copy()
    _check_fitted(ModK2)
    assert np.isclose(ModK2.cluster_centers_, ModK1.cluster_centers_).all()
    assert np.isclose(ModK2.GEV_, ModK1.GEV_)
    assert np.isclose(ModK2.labels_, ModK1.labels_).all()
    ModK2.fitted = False
    _check_fitted(ModK1)
    _check_unfitted(ModK2)

    ModK3 = ModK1.copy(deep=False)
    _check_fitted(ModK3)
    assert np.isclose(ModK3.cluster_centers_, ModK1.cluster_centers_).all()
    assert np.isclose(ModK3.GEV_, ModK1.GEV_)
    assert np.isclose(ModK3.labels_, ModK1.labels_).all()
    ModK3.fitted = False
    _check_fitted(ModK1)
    _check_unfitted(ModK3)

    # Test representation
    expected = f'<ModKMeans | fitted on n = {n_clusters} cluster centers>'
    assert expected == ModK1.__repr__()
    assert '<ModKMeans | not fitted>' == ModK2.__repr__()

    # Test plot
    f, ax = ModK1.plot(block=False)
    with pytest.raises(RuntimeError, match='must be fitted before'):
        f, ax = ModK2.plot(block=False)
    plt.close('all')


def test_invert_polarity():
    """Test invert polarity method."""
    # list/tuple
    ModK_ = ModK.copy()
    cluster_centers_ = deepcopy(ModK_.cluster_centers_)
    ModK_.invert_polarity([True, False, True, False])
    assert np.isclose(
        ModK_.cluster_centers_[0, :], -cluster_centers_[0, :]).all()
    assert np.isclose(
        ModK_.cluster_centers_[1, :], cluster_centers_[1, :]).all()
    assert np.isclose(
        ModK_.cluster_centers_[2, :], -cluster_centers_[2, :]).all()
    assert np.isclose(
        ModK_.cluster_centers_[3, :], cluster_centers_[3, :]).all()

    # bool
    ModK_ = ModK.copy()
    cluster_centers_ = deepcopy(ModK_.cluster_centers_)
    ModK_.invert_polarity(True)
    assert np.isclose(
        ModK_.cluster_centers_[0, :], -cluster_centers_[0, :]).all()
    assert np.isclose(
        ModK_.cluster_centers_[1, :], -cluster_centers_[1, :]).all()
    assert np.isclose(
        ModK_.cluster_centers_[2, :], -cluster_centers_[2, :]).all()
    assert np.isclose(
        ModK_.cluster_centers_[3, :], -cluster_centers_[3, :]).all()

    # np.array
    ModK_ = ModK.copy()
    cluster_centers_ = deepcopy(ModK_.cluster_centers_)
    ModK_.invert_polarity(np.array([True, False, True, False]))
    assert np.isclose(
        ModK_.cluster_centers_[0, :], -cluster_centers_[0, :]).all()
    assert np.isclose(
        ModK_.cluster_centers_[1, :], cluster_centers_[1, :]).all()
    assert np.isclose(
        ModK_.cluster_centers_[2, :], -cluster_centers_[2, :]).all()
    assert np.isclose(
        ModK_.cluster_centers_[3, :], cluster_centers_[3, :]).all()

    # Test invalid arguments
    with pytest.raises(ValueError, match="not a 2D iterable"):
        ModK_.invert_polarity(np.zeros((2, 4)))
    with pytest.raises(
            ValueError,
            match=re.escape("list of bools of length 'n_clusters' (4)")):
        ModK_.invert_polarity([True, False, True, False, True])
    with pytest.raises(TypeError, match="'invert' must be an instance of "):
        ModK_.invert_polarity(101)

    # Test unfitted
    ModK_.fitted = False
    _check_unfitted(ModK_)
    with pytest.raises(RuntimeError, match='must be fitted before'):
        ModK_.invert_polarity([True, False, True, False])


def test_rename(caplog):
    """Test renaming of clusters."""
    alphabet = ['A', 'B', 'C', 'D']

    # Test mapping
    ModK_ = ModK.copy()
    mapping = {old: alphabet[k] for k, old in enumerate(ModK.clusters_names)}
    for key, value in mapping.items():
        assert isinstance(key, str)
        assert isinstance(value, str)
        assert key != value
    ModK_.rename_clusters(mapping=mapping)
    assert ModK_.clusters_names == alphabet
    assert ModK_.clusters_names != ModK.clusters_names

    # Test new_names
    ModK_ = ModK.copy()
    ModK_.rename_clusters(new_names=alphabet)
    assert ModK_.clusters_names == alphabet
    assert ModK_.clusters_names != ModK.clusters_names

    # Test invalid arguments
    ModK_ = ModK.copy()
    with pytest.raises(TypeError, match="'mapping' must be an instance of "):
        ModK_.rename_clusters(mapping=101)
    with pytest.raises(ValueError, match="Invalid value for the 'old name'"):
        mapping = {old + '101': alphabet[k]
                   for k, old in enumerate(ModK.clusters_names)}
        ModK_.rename_clusters(mapping=mapping)
    with pytest.raises(TypeError, match="'new name' must be an instance of "):
        mapping = {old: k for k, old in enumerate(ModK.clusters_names)}
        ModK_.rename_clusters(mapping=mapping)
    with pytest.raises(ValueError,
                       match="Argument 'new_names' should contain"):
        ModK_.rename_clusters(new_names=alphabet+['E'])

    ModK_.rename_clusters()
    assert "Either 'mapping' or 'new_names' should not be" in caplog.text

    with pytest.raises(ValueError,
                       match="Only one of 'mapping' or 'new_names'"):
        mapping = {old: alphabet[k] for
                   k, old in enumerate(ModK.clusters_names)}
        ModK_.rename_clusters(mapping=mapping, new_names=alphabet)

    # Test unfitted
    ModK_ = ModK.copy()
    ModK_.fitted = False
    _check_unfitted(ModK_)
    with pytest.raises(RuntimeError, match='must be fitted before'):
        mapping = {old: alphabet[k]
                   for k, old in enumerate(ModK.clusters_names)}
        ModK_.rename_clusters(mapping=mapping)
    with pytest.raises(RuntimeError, match='must be fitted before'):
        ModK_.rename_clusters(new_names=alphabet)


def test_reorder(caplog):
    """Test reordering of clusters."""
    # Test mapping
    ModK_ = ModK.copy()
    ModK_.reorder_clusters(mapping={0: 1})
    assert np.isclose(
        ModK.cluster_centers_[0, :], ModK_.cluster_centers_[1, :]).all()
    assert np.isclose(
        ModK.cluster_centers_[1, :], ModK_.cluster_centers_[0, :]).all()
    assert ModK.clusters_names[0] == ModK_.clusters_names[1]
    assert ModK.clusters_names[0] == ModK_.clusters_names[1]

    # Test order
    ModK_ = ModK.copy()
    ModK_.reorder_clusters(order=[1, 0, 2, 3])
    assert np.isclose(
        ModK.cluster_centers_[0], ModK_.cluster_centers_[1]).all()
    assert np.isclose(
        ModK.cluster_centers_[1], ModK_.cluster_centers_[0]).all()
    assert ModK.clusters_names[0] == ModK_.clusters_names[1]
    assert ModK.clusters_names[0] == ModK_.clusters_names[1]

    ModK_ = ModK.copy()
    ModK_.reorder_clusters(order=np.array([1, 0, 2, 3]))
    assert np.isclose(
        ModK.cluster_centers_[0], ModK_.cluster_centers_[1]).all()
    assert np.isclose(
        ModK.cluster_centers_[1], ModK_.cluster_centers_[0]).all()
    assert ModK.clusters_names[0] == ModK_.clusters_names[1]
    assert ModK.clusters_names[0] == ModK_.clusters_names[1]

    # test .labels_ reordering
    x = ModK_.labels_[:20]
    # x: before re-order:
    # x = [3, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    # y: expected re-ordered labels
    y = [3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    assert np.all(x == y)

    # Test invalid arguments
    ModK_ = ModK.copy()
    with pytest.raises(TypeError, match="'mapping' must be an instance of "):
        ModK_.reorder_clusters(mapping=101)
    with pytest.raises(ValueError,
                       match="Invalid value for the 'old position'"):
        ModK_.reorder_clusters(mapping={4: 1})
    with pytest.raises(ValueError,
                       match="Invalid value for the 'new position'"):
        ModK_.reorder_clusters(mapping={0: 4})
    with pytest.raises(ValueError,
                       match='Position in the new order can not be repeated.'):
        ModK_.reorder_clusters(mapping={0: 1, 2: 1})
    with pytest.raises(ValueError,
                       match='A position can not be present in both'):
        ModK_.reorder_clusters(mapping={0: 1, 1: 2})

    with pytest.raises(TypeError, match="'order' must be an instance of "):
        ModK_.reorder_clusters(order=101)
    with pytest.raises(ValueError, match="Invalid value for the 'order'"):
        ModK_.reorder_clusters(order=[4, 3, 1, 2])
    with pytest.raises(ValueError,
                       match="Argument 'order' should contain 'n_clusters'"):
        ModK_.reorder_clusters(order=[0, 3, 1, 2, 0])
    with pytest.raises(ValueError,
                       match="Argument 'order' should be a 1D iterable"):
        ModK_.reorder_clusters(order=np.array([[0, 1, 2, 3], [0, 1, 2, 3]]))

    ModK_.reorder_clusters()
    assert "Either 'mapping' or 'order' should not be 'None' " in caplog.text

    with pytest.raises(ValueError,
                       match="Only one of 'mapping' or 'order'"):
        ModK_.reorder_clusters(mapping={0: 1}, order=[1, 0, 2, 3])

    # Test unfitted
    ModK_ = ModK.copy()
    ModK_.fitted = False
    _check_unfitted(ModK_)
    with pytest.raises(RuntimeError, match='must be fitted before'):
        ModK_.reorder_clusters(mapping={0: 1})
    with pytest.raises(RuntimeError, match='must be fitted before'):
        ModK_.reorder_clusters(order=[1, 0, 2, 3])


def test_properties(caplog):
    """Test properties."""
    caplog.set_level(logging.WARNING)

    # Unfitted
    ModK_ = ModKMeans(n_clusters=n_clusters, n_init=10, max_iter=100, tol=1e-4,
                      random_state=1)

    ModK_.cluster_centers_
    assert 'Clustering algorithm has not been fitted.' in caplog.text
    caplog.clear()

    ModK_.picks
    assert 'Clustering algorithm has not been fitted.' in caplog.text
    caplog.clear()

    ModK_.info
    assert 'Clustering algorithm has not been fitted.' in caplog.text
    caplog.clear()

    ModK_.fitted_data
    assert 'Clustering algorithm has not been fitted.' in caplog.text
    caplog.clear()

    # Fitted
    ModK_ = ModK.copy()

    ModK_.cluster_centers_
    assert 'Clustering algorithm has not been fitted.' not in caplog.text
    caplog.clear()

    ModK_.picks
    assert 'Clustering algorithm has not been fitted.' not in caplog.text
    caplog.clear()

    ModK_.info
    assert 'Clustering algorithm has not been fitted.' not in caplog.text
    caplog.clear()

    ModK_.fitted_data
    assert 'Clustering algorithm has not been fitted.' not in caplog.text
    caplog.clear()

    # Test fitted property
    ModK_ = ModKMeans(n_clusters=n_clusters, n_init=10, max_iter=100, tol=1e-4,
                      random_state=1)
    with pytest.raises(TypeError, match="'fitted' must be an instance of"):
        ModK_.fitted = '101'
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
    with pytest.raises(TypeError,
                       match="'n_clusters' must be an instance of "):
        ModK_ = ModKMeans(n_clusters='4')
    with pytest.raises(ValueError, match="The number of clusters must be a"):
        ModK_ = ModKMeans(n_clusters=0)
    with pytest.raises(ValueError, match="The number of clusters must be a"):
        ModK_ = ModKMeans(n_clusters=-101)

    # n_init
    with pytest.raises(TypeError, match="'n_init' must be an instance of "):
        ModK_ = ModKMeans(n_clusters=4, n_init='100')
    with pytest.raises(ValueError,
                       match="The number of initialization must be a"):
        ModK_ = ModKMeans(n_clusters=4, n_init=0)
    with pytest.raises(ValueError,
                       match="The number of initialization must be a"):
        ModK_ = ModKMeans(n_clusters=4, n_init=-101)

    # max_iter
    with pytest.raises(TypeError, match="'max_iter' must be an instance of "):
        ModK_ = ModKMeans(n_clusters=4, max_iter='100')
    with pytest.raises(ValueError,
                       match="The number of max iteration must be a"):
        ModK_ = ModKMeans(n_clusters=4, max_iter=0)
    with pytest.raises(ValueError,
                       match="The number of max iteration must be a"):
        ModK_ = ModKMeans(n_clusters=4, max_iter=-101)

    # tol
    with pytest.raises(TypeError, match="'tol' must be an instance of "):
        ModK_ = ModKMeans(n_clusters=4, tol='100')
    with pytest.raises(ValueError, match="The tolerance must be a"):
        ModK_ = ModKMeans(n_clusters=4, tol=0)
    with pytest.raises(ValueError, match="The tolerance must be a"):
        ModK_ = ModKMeans(n_clusters=4, tol=-101)

    # random-state
    with pytest.raises(ValueError, match="cannot be used to seed"):
        ModK_ = ModKMeans(n_clusters=4, random_state='101')

    ModK_ = ModKMeans(n_clusters=n_clusters, n_init=10, max_iter=100, tol=1e-4,
                      random_state=1)
    # inst
    with pytest.raises(TypeError, match="'inst' must be an instance of "):
        ModK_.fit(epochs.average())

    # tmin/tmax
    with pytest.raises(TypeError, match="'tmin' must be an instance of "):
        ModK_.fit(raw, tmin='101')
    with pytest.raises(TypeError, match="'tmax' must be an instance of "):
        ModK_.fit(raw, tmax='101')
    with pytest.raises(ValueError, match="Argument 'tmin' must be positive"):
        ModK_.fit(raw, tmin=-101, tmax=None)
    with pytest.raises(ValueError, match="Argument 'tmax' must be positive"):
        ModK_.fit(raw, tmin=None, tmax=-101)
    with pytest.raises(
            ValueError,
            match="Argument 'tmax' must be strictly larger than 'tmin'."):
        ModK_.fit(raw, tmin=5, tmax=1)
    with pytest.raises(
            ValueError,
            match="Argument 'tmin' must be shorter than the instance length."):
        ModK_.fit(raw, tmin=101, tmax=None)
    with pytest.raises(
            ValueError,
            match="Argument 'tmax' must be shorter than the instance length."):
        ModK_.fit(raw, tmin=None, tmax=101)

    # reject_by_annotation
    with pytest.raises(TypeError,
                       match="'reject_by_annotation' must be an instance of "):
        ModK_.fit(raw, reject_by_annotation=1)
    with pytest.raises(ValueError, match='only allows for'):
        ModK_.fit(raw, reject_by_annotation='101')


def test_fit_data_shapes():
    """Test different tmin/tmax, rejection with fit."""
    ModK_ = ModKMeans(n_clusters=n_clusters, n_init=10, max_iter=100, tol=1e-4,
                      random_state=1)

    # tmin
    ModK_.fitted = False
    _check_unfitted(ModK_)
    ModK_.fit(raw, n_jobs=1, picks='eeg', tmin=5, tmax=None,
              reject_by_annotation=False)
    _check_fitted_data_raw(ModK_.fitted_data, raw, 'eeg', 5, None, None)
    # save for later
    fitted_data_5_end = deepcopy(ModK_.fitted_data)

    ModK_.fitted = False
    _check_unfitted(ModK_)
    ModK_.fit(epochs, n_jobs=1, picks='eeg', tmin=0.2, tmax=None,
              reject_by_annotation=False)
    _check_fitted_data_epochs(ModK_.fitted_data, epochs, 'eeg', 0.2, None)

    # tmax
    ModK_.fitted = False
    _check_unfitted(ModK_)
    ModK_.fit(raw, n_jobs=1, picks='eeg', tmin=None, tmax=5,
              reject_by_annotation=False)
    _check_fitted_data_raw(ModK_.fitted_data, raw, 'eeg', None, 5, None)
    # save for later
    fitted_data_0_5 = deepcopy(ModK_.fitted_data)

    ModK_.fitted = False
    _check_unfitted(ModK_)
    ModK_.fit(epochs, n_jobs=1, picks='eeg', tmin=None, tmax=0.3,
              reject_by_annotation=False)
    _check_fitted_data_epochs(ModK_.fitted_data, epochs, 'eeg', None, 0.3)

    # tmin, tmax
    ModK_.fitted = False
    _check_unfitted(ModK_)
    ModK_.fit(raw, n_jobs=1, picks='eeg', tmin=2, tmax=8,
              reject_by_annotation=False)
    _check_fitted_data_raw(ModK_.fitted_data, raw, 'eeg', 2, 8, None)

    ModK_.fitted = False
    _check_unfitted(ModK_)
    ModK_.fit(epochs, n_jobs=1, picks='eeg', tmin=0.1, tmax=0.4,
              reject_by_annotation=False)
    _check_fitted_data_epochs(ModK_.fitted_data, epochs, 'eeg', 0.1, 0.4)

    # ---------------------
    # Reject by annotations
    # ---------------------
    bad_annot = mne.Annotations([1], [2], 'bad')
    raw_ = raw.copy()
    raw_.set_annotations(bad_annot)

    ModK_.fitted = False
    _check_unfitted(ModK_)

    ModK_no_reject = ModK_.copy()
    ModK_no_reject.fit(raw_, n_jobs=1, reject_by_annotation=False)
    ModK_reject_True = ModK_.copy()
    ModK_reject_True.fit(raw_, n_jobs=1, reject_by_annotation=True)
    ModK_reject_omit = ModK_.copy()
    ModK_reject_omit.fit(raw_, n_jobs=1, reject_by_annotation='omit')

    # Compare 'omit' and True
    assert np.isclose(ModK_reject_omit.fitted_data,
                      ModK_reject_True.fitted_data).all()
    assert np.isclose(ModK_reject_omit.GEV_, ModK_reject_True.GEV_)
    assert np.isclose(ModK_reject_omit.labels_, ModK_reject_True.labels_).all()
    assert np.isclose(ModK_reject_omit.cluster_centers_,
                      ModK_reject_True.cluster_centers_).all()

    # Make sure there is a shape diff between True and False
    assert ModK_reject_True.fitted_data.shape != \
        ModK_no_reject.fitted_data.shape

    # Check fitted data
    _check_fitted_data_raw(
        ModK_reject_True.fitted_data, raw_, 'eeg', None, None, 'omit')
    _check_fitted_data_raw(
        ModK_no_reject.fitted_data, raw_, 'eeg', None, None, None)

    # Check with reject with tmin/tmax
    ModK_rej_0_5 = ModK_.copy()
    ModK_rej_0_5.fit(raw_, n_jobs=1, tmin=0, tmax=5, reject_by_annotation=True)
    ModK_rej_5_end = ModK_.copy()
    ModK_rej_5_end.fit(raw_, n_jobs=1, tmin=5, tmax=None,
                       reject_by_annotation=True)
    _check_fitted(ModK_rej_0_5)
    _check_fitted(ModK_rej_5_end)
    _check_fitted_data_raw(
        ModK_rej_0_5.fitted_data, raw_, 'eeg', None, 5, 'omit')
    _check_fitted_data_raw(
        ModK_rej_5_end.fitted_data, raw_, 'eeg', 5, None, 'omit')
    assert ModK_rej_0_5.fitted_data.shape != fitted_data_0_5.shape
    assert np.isclose(fitted_data_5_end, ModK_rej_5_end.fitted_data).all()


def test_fit_with_bads(caplog):
    """Test log messages emitted when fitting with bad channels."""
    # 0 bads
    raw_ = raw.copy()
    raw_.info['bads'] = list()
    ModK_ = ModKMeans(n_clusters=n_clusters, n_init=10, max_iter=100, tol=1e-4,
                      random_state=1)
    ModK_.fit(raw_, n_jobs=1)
    _check_fitted(ModK_)
    _check_fitted_data_raw(ModK_.fitted_data, raw_, 'eeg', None, None, 'omit')
    assert len(ModK_.info['bads']) == 0
    assert len(ModK_.picks) == 60
    assert ModK_.picks[0] == 0
    assert 'set as bad' not in caplog.text
    caplog.clear()

    # 1 bads
    raw_ = raw.copy()
    raw_.info['bads'] = [raw_.ch_names[0]]
    ModK_ = ModKMeans(n_clusters=n_clusters, n_init=10, max_iter=100, tol=1e-4,
                      random_state=1)
    ModK_.fit(raw_, n_jobs=1)
    _check_fitted(ModK_)
    _check_fitted_data_raw(ModK_.fitted_data, raw_, 'eeg', None, None, 'omit')
    assert len(ModK_.info['bads']) == 0  # pick_info on picks
    assert len(ModK_.picks) == 59
    assert ModK_.picks[0] == 1
    assert 'Channel EEG 001 is set as bad and ignored.' in caplog.text
    caplog.clear()

    # more than 1 bads
    raw_ = raw.copy()
    raw_.info['bads'] = [raw_.ch_names[0], raw_.ch_names[1]]
    ModK_ = ModKMeans(n_clusters=n_clusters, n_init=10, max_iter=100, tol=1e-4,
                      random_state=1)
    ModK_.fit(raw_, n_jobs=1)
    _check_fitted(ModK_)
    _check_fitted_data_raw(ModK_.fitted_data, raw_, 'eeg', None, None, 'omit')
    assert len(ModK_.info['bads']) == 0  # pick_info on picks
    assert len(ModK_.picks) == 58
    assert ModK_.picks[0] == 2
    assert 'Channels EEG 001, EEG 002 are set as bads' in caplog.text
    caplog.clear()

    # Test on epochs
    epochs_ = epochs.copy()
    epochs_.info['bads'] = [epochs_.ch_names[0], epochs_.ch_names[1]]
    ModK_ = ModKMeans(n_clusters=n_clusters, n_init=10, max_iter=100, tol=1e-4,
                      random_state=1)
    ModK_.fit(epochs_, n_jobs=1)
    _check_fitted(ModK_)
    _check_fitted_data_epochs(ModK_.fitted_data, epochs_, 'eeg', None, None)
    assert len(ModK_.info['bads']) == 0  # pick_info on picks
    assert len(ModK_.picks) == 58
    assert ModK_.picks[0] == 2
    assert 'Channels EEG 001, EEG 002 are set as bads' in caplog.text
    caplog.clear()


def test_predict(caplog):
    """Test predict method default behaviors."""
    # raw, no smoothing, no_edge
    segmentation = ModK.predict(raw, factor=0, reject_edges=False)
    assert isinstance(segmentation, RawSegmentation)
    assert 'Segmenting data without smoothing' in caplog.text
    caplog.clear()

    # raw, no smoothing, with edge rejection
    segmentation = ModK.predict(raw, factor=0, reject_edges=True)
    assert isinstance(segmentation, RawSegmentation)
    assert segmentation.labels[0] == -1
    assert segmentation.labels[-1] == -1
    assert 'Rejecting first and last segments.' in caplog.text
    caplog.clear()

    # raw, with smoothing
    segmentation = ModK.predict(raw, factor=3, reject_edges=True)
    assert isinstance(segmentation, RawSegmentation)
    assert segmentation.labels[0] == -1
    assert segmentation.labels[-1] == -1
    assert 'Segmenting data with factor 3' in caplog.text
    caplog.clear()

    # raw with min_segment_length
    segmentation = ModK.predict(raw, factor=0, reject_edges=False,
                                min_segment_length=5)
    assert isinstance(segmentation, RawSegmentation)
    segment_lengths = [len(list(group))
                       for _, group in groupby(segmentation.labels)]
    assert all(5 <= size for size in segment_lengths[1:-1])
    assert 'Rejecting segments shorter than' in caplog.text
    caplog.clear()

    # epochs, no smoothing, no_edge
    segmentation = ModK.predict(epochs, factor=0, reject_edges=False)
    assert isinstance(segmentation, EpochsSegmentation)
    assert 'Segmenting data without smoothing' in caplog.text
    caplog.clear()

    # epochs, no smoothing, with edge rejection
    segmentation = ModK.predict(epochs, factor=0, reject_edges=True)
    assert isinstance(segmentation, EpochsSegmentation)
    for epoch_labels in segmentation.labels:
        assert epoch_labels[0] == -1
        assert epoch_labels[-1] == -1
    assert 'Rejecting first and last segments.' in caplog.text
    caplog.clear()

    # epochs, with smoothing
    segmentation = ModK.predict(epochs, factor=3, reject_edges=True)
    assert isinstance(segmentation, EpochsSegmentation)
    for epoch_labels in segmentation.labels:
        assert epoch_labels[0] == -1
        assert epoch_labels[-1] == -1
    assert 'Segmenting data with factor 3' in caplog.text
    caplog.clear()

    # epochs with min_segment_length
    segmentation = ModK.predict(epochs, factor=0, reject_edges=False,
                                min_segment_length=5)
    assert isinstance(segmentation, EpochsSegmentation)
    for epoch_labels in segmentation.labels:
        segment_lengths = [len(list(group))
                           for _, group in groupby(epoch_labels)]
        assert all(5 <= size for size in segment_lengths[1:-1])
    assert 'Rejecting segments shorter than' in caplog.text
    caplog.clear()

    # raw with reject_by_annotation
    bad_annot = mne.Annotations([1], [2], 'bad')
    raw_ = raw.copy()
    raw_.set_annotations(bad_annot)
    segmentation_rej_True = ModK.predict(raw_, factor=0, reject_edges=True,
                                         reject_by_annotation=True)
    segmentation_rej_False = ModK.predict(raw_, factor=0, reject_edges=True,
                                          reject_by_annotation=False)
    segmentation_rej_None = ModK.predict(raw_, factor=0, reject_edges=True,
                                         reject_by_annotation=None)
    segmentation_no_annot = ModK.predict(raw, factor=0, reject_edges=True,
                                         reject_by_annotation='omit')
    assert not np.isclose(segmentation_rej_True.labels,
                          segmentation_rej_False.labels).all()
    assert np.isclose(segmentation_no_annot.labels,
                      segmentation_rej_False.labels).all()
    assert np.isclose(segmentation_rej_None.labels,
                      segmentation_rej_False.labels).all()

    # test different half_window_size
    segmentation1 = ModK.predict(raw, factor=3, reject_edges=False,
                                 half_window_size=3)
    segmentation2 = ModK.predict(raw, factor=3, reject_edges=False,
                                 half_window_size=60)
    segmentation3 = ModK.predict(raw, factor=0, reject_edges=False,
                                 half_window_size=3)
    assert not np.isclose(segmentation1.labels,
                          segmentation2.labels).all()
    assert not np.isclose(segmentation1.labels,
                          segmentation3.labels).all()
    assert not np.isclose(segmentation2.labels,
                          segmentation3.labels).all()


def test_predict_invalid_arguments(caplog):
    """Test invalid arguments passed to predict."""
    with pytest.raises(TypeError, match="'inst' must be an instance of "):
        ModK.predict(epochs.average())
    with pytest.raises(TypeError, match="'factor' must be an instance of "):
        ModK.predict(raw, factor='0')
    with pytest.raises(TypeError,
                       match="'reject_edges' must be an instance of "):
        ModK.predict(raw, reject_edges=1)
    with pytest.raises(TypeError,
                       match="'half_window_size' must be an instance of "):
        ModK.predict(raw, half_window_size='1')
    with pytest.raises(TypeError, match="'tol' must be an instance of "):
        ModK.predict(raw, tol='0')
    with pytest.raises(TypeError,
                       match="'min_segment_length' must be an instance of "):
        ModK.predict(raw, min_segment_length='0')
    with pytest.raises(TypeError,
                       match="'reject_by_annotation' must be an instance of "):
        ModK.predict(raw, reject_by_annotation=1)

    # TODO: Shouldn't that be a raise ValueError?
    caplog.clear()
    ModK.predict(raw, reject_by_annotation='101')
    assert "'reject_by_annotation' can be set to" in caplog.text
    caplog.clear()

    raw_ = raw.copy().drop_channels([raw.ch_names[0]])
    with pytest.raises(ValueError, match='does not have the same channels'):
        ModK.predict(raw_)


def test_n_jobs():
    """Test that n_jobs=2 works."""
    ModK_ = ModKMeans(n_clusters=n_clusters, n_init=10, max_iter=100, tol=1e-4,
                      random_state=1)
    ModK_.fit(raw, n_jobs=2)
    _check_fitted(ModK_)
    assert np.isclose(ModK_.cluster_centers_, ModK.cluster_centers_).all()
    assert np.isclose(ModK_.GEV_, ModK.GEV_)
    assert np.isclose(ModK_.labels_, ModK.labels_).all()


def test_fit_not_converged(caplog):
    """Test a fit that did not converge."""
    # 10/10 converged
    ModK_ = ModKMeans(n_clusters=n_clusters, n_init=10, max_iter=40, tol=1e-4,
                      random_state=1)
    ModK_.fit(raw, n_jobs=1)
    _check_fitted(ModK_)
    assert 'after 10/10 iterations converged.' in caplog.text
    caplog.clear()

    # 6/10 converged
    ModK_ = ModKMeans(n_clusters=n_clusters, n_init=10, max_iter=20, tol=1e-10,
                      random_state=1)
    ModK_.fit(raw, n_jobs=1)
    _check_fitted(ModK_)
    assert 'after 6/10 iterations converged.' in caplog.text
    caplog.clear()

    # 0/10 converged
    ModK_ = ModKMeans(n_clusters=n_clusters, n_init=10, max_iter=10, tol=1e-20,
                      random_state=1)
    ModK_.fit(raw, n_jobs=1)
    _check_unfitted(ModK_)
    assert "All the K-means run failed to converge." in caplog.text
    caplog.clear()

    ModK_ = ModKMeans(n_clusters=n_clusters, n_init=10, max_iter=10, tol=1e-20,
                      random_state=1)
    ModK_.fit(raw, n_jobs=2)
    _check_unfitted(ModK_)
    assert "All the K-means run failed to converge." in caplog.text
    caplog.clear()


def test_randomseed():
    """Test that fit with same randomseed provide the same output."""
    ModK1 = ModKMeans(n_clusters=n_clusters, n_init=10, max_iter=40, tol=1e-4,
                      random_state=1)
    ModK1.fit(raw, n_jobs=1)
    ModK2 = ModKMeans(n_clusters=n_clusters, n_init=10, max_iter=40, tol=1e-4,
                      random_state=1)
    ModK2.fit(raw, n_jobs=1)
    ModK3 = ModKMeans(n_clusters=n_clusters, n_init=10, max_iter=40, tol=1e-4,
                      random_state=2)
    ModK3.fit(raw, n_jobs=1)

    assert np.isclose(ModK1.cluster_centers_, ModK2.cluster_centers_).all()
    assert not np.isclose(ModK1.cluster_centers_, ModK3.cluster_centers_).all()


def test_contains_mixin():
    """Test mixin classes."""
    assert 'eeg' in ModK
    assert ModK.compensation_grade is None
    assert ModK.get_channel_types() == ['eeg'] * ModK.info['nchan']
