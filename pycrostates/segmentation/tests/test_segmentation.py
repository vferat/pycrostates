from pathlib import Path

import matplotlib.pyplot as plt
import mne
from mne import Epochs
from mne.datasets import testing
from mne.io import BaseRaw
import numpy as np
import pytest

from pycrostates.cluster import ModKMeans
from pycrostates.segmentation import RawSegmentation, EpochsSegmentation
from pycrostates.utils._logs import logger, set_log_level


set_log_level('INFO')
logger.propagate = True


dir_ = Path(testing.data_path()) / 'MEG' / 'sample'
fname_raw_testing = dir_ / 'sample_audvis_trunc_raw.fif'
raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
raw = raw.pick('eeg').crop(0, 10).filter(0, 40).apply_proj()

events = mne.make_fixed_length_events(raw, 1)
epochs = mne.epochs.Epochs(raw, events, preload=True)

ModK_raw = ModKMeans(n_clusters=4, n_init=10, max_iter=100, tol=1e-4,
                     random_state=1)
ModK_epochs = ModKMeans(n_clusters=4, n_init=10, max_iter=100, tol=1e-4,
                        random_state=1)
ModK_raw.fit(raw, n_jobs=1)
ModK_epochs.fit(epochs, n_jobs=1)


# pylint: disable=protected-access
@pytest.mark.parametrize('ModK, inst', [
    (ModK_raw, raw),
    (ModK_epochs, epochs),
    ])
def test_properties(ModK, inst, caplog):
    """Test properties from segmentation."""
    segmentation = ModK.predict(inst)
    type_ = BaseRaw if isinstance(inst, BaseRaw) else Epochs
    assert isinstance(segmentation._inst, type_)
    assert isinstance(segmentation._cluster_centers_, np.ndarray)
    assert isinstance(segmentation._cluster_names, list)
    assert isinstance(segmentation._labels, np.ndarray)
    ndim = 1 if isinstance(inst, BaseRaw) else 2
    assert segmentation._labels.ndim == ndim
    assert isinstance(segmentation._predict_parameters, dict)

    # test that we do get copies
    cluster_centers_ = segmentation.cluster_centers_
    cluster_names = segmentation.cluster_names
    labels = segmentation.labels
    predict_parameters = segmentation.predict_parameters

    assert isinstance(cluster_centers_, np.ndarray)
    assert isinstance(cluster_names, list)
    assert isinstance(labels, np.ndarray)
    assert labels.ndim == 1
    assert isinstance(predict_parameters, dict)

    cluster_centers_ -= 10
    assert cluster_centers_ == segmentation._cluster_centers_ - 10
    labels -= 10
    assert labels == segmentation._labels - 10
    predict_parameters['test'] = 10
    assert 'test' not in segmentation._predict_parameters

    # test raw/epochs specific
    if isinstance(inst, BaseRaw):
        raw_ = segmentation.raw
        assert isinstance(raw_, BaseRaw)
        raw_.drop_channels(raw_.ch_names[:3])
        assert raw_.ch_names == segmentation._inst.ch_names[3:]
    if isinstance(inst, Epochs):
        epochs_ = segmentation.epochs
        assert isinstance(epochs_, Epochs)
        epochs_.drop_channels(raw_.ch_names[:3])
        assert epochs_.ch_names == segmentation._inst.ch_names[3:]

    # test without predict_parameters
    segmentation = ModK.predict(inst)
    segmentation._predict_parameters = None
    caplog.clear()
    params = segmentation.predict_parameters
    assert params is None
    assert "predict_parameters' was noprovided when creating" in caplog.text

    # test that properties can not be set
    segmentation = ModK.predict(inst)
    with pytest.raises(AttributeError, match="can't set attribute"):
        segmentation.predict_parameters = dict()
    with pytest.raises(AttributeError, match="can't set attribute"):
        segmentation.labels = np.zeros((raw.times.size, ))
    with pytest.raises(AttributeError, match="can't set attribute"):
        segmentation.cluster_names = ['1', '0', '2', '3']
    with pytest.raises(AttributeError, match="can't set attribute"):
        segmentation.cluster_centers_ = \
            np.zeros((4, len(inst.ch_names) - len(inst.info['bads'])))

    # test raw/epochs specific
    with pytest.raises(AttributeError, match="can't set attribute"):
        if isinstance(inst, BaseRaw):
            segmentation.raw = raw
        if isinstance(inst, Epochs):
            segmentation.epochs = epochs


@pytest.mark.parametrize('ModK, inst', [
    (ModK_raw, raw),
    (ModK_epochs, epochs),
    ])
def test_plot_cluster_centers(ModK, inst):
    """Test plot_cluster_centers method."""
    # with raw
    segmentation = ModK.predict(inst)
    segmentation.plot_cluster_centers()
    plt.close('all')

    # with axes provided
    f, ax = plt.subplots(2, 2)
    segmentation.plot_cluster_centers(axes=ax)
    plt.close('all')


@pytest.mark.parametrize('ModK, inst', [
    (ModK_raw, raw),
    (ModK_epochs, epochs),
    ])
def test_compute_parameters(ModK, inst):
    """Test compute_parameters method."""
    segmentation = ModK.predict(inst)
    params = segmentation.compute_parameters(norm_gfp=False)
    assert isinstance(params, dict)
    assert '1_dist_corr' not in params.keys()

    # with normalization of GFP
    params = segmentation.compute_parameters(norm_gfp=True)
    assert isinstance(params, dict)

    # with return_dist
    params = segmentation.compute_parameters(norm_gfp=False, return_dist=True)
    assert isinstance(params, dict)
    assert '1_dist_corr' in params.keys()

    # with invalid types for norm_gfp or return_dist
    with pytest.raises(TypeError, match='must be an instance of'):
        segmentation.compute_parameters(norm_gfp=1)
    with pytest.raises(TypeError, match='must be an instance of'):
        segmentation.compute_parameters(return_dist=1)


@pytest.mark.parametrize('ModK, inst', [
    (ModK_raw, raw),
    (ModK_epochs, epochs),
    ])
def test_repr(ModK, inst):
    """Test for representations of segmentation."""
    segmentation = ModK.predict(inst)
    assert segmentation.__repr__() is not None
    assert isinstance(segmentation.__repr__(), str)
    assert segmentation._repr_html_() is not None


@pytest.mark.parametrize('ModK, inst', [
    (ModK_raw, raw),
    (ModK_epochs, epochs),
    ])
def test_plot_segmentation(ModK, inst):
    """Test the plot of a segmentation."""
    segmentation = ModK.predict(inst)

    segmentation.plot()
    plt.close('all')
    segmentation.plot(cmap='plasma')
    plt.close('all')
    f, ax = plt.subplots(1, 1)
    segmentation.plot(axes=ax)
    plt.close('all')
    f, ax = plt.subplots(1, 1)
    segmentation.plot(cbar_axes=ax)
    plt.close('all')

    # specific to raw
    if isinstance(inst, BaseRaw):
        segmentation.plot(tmin=0, tmax=5)
        plt.close('all')


@pytest.mark.parametrize('ModK, inst, bad_inst', [
    (RawSegmentation, raw, epochs),
    (EpochsSegmentation, epochs, raw),
    ])
def test_invalid_segmentation(Segmentation, inst, bad_inst, caplog):
    """Test that we can not create an invalid segmentation."""
    labels = np.zeros((inst.times.size))
    cluster_centers = \
        np.zeros((4, len(inst.ch_names) - len(inst.info['bads'])))
    cluster_names = ['a', 'b', 'c', 'd']

    # types
    with pytest.raises(TypeError, match='must be an instance of'):
        Segmentation(list(labels), inst, cluster_centers, cluster_names, None)
    with pytest.raises(TypeError, match='must be an instance of'):
        Segmentation(labels, 101, cluster_centers, cluster_names, None)
    with pytest.raises(TypeError, match='must be an instance of'):
        Segmentation(labels, bad_inst, cluster_centers, cluster_names, None)
    with pytest.raises(TypeError, match='must be an instance of'):
        Segmentation(labels, inst, list(cluster_centers), cluster_names, None)
    with pytest.raises(TypeError, match='must be an instance of'):
        Segmentation(labels, inst, cluster_centers, tuple(cluster_names), None)
    with pytest.raises(TypeError, match='must be an instance of'):
        Segmentation(labels, inst, cluster_centers, cluster_names, [])

    # values
    with pytest.raises(ValueError,
                       match='number of cluster centers and cluster names'):
        Segmentation(labels, inst, cluster_centers, cluster_names[:2], None)
    with pytest.raises(ValueError, match='should be a 2D array'):
        Segmentation(labels, inst, cluster_centers.flatten(), cluster_names,
                     None)
    with pytest.raises(ValueError, match='instance and labels do not have'):
        Segmentation(labels[:10], inst, cluster_centers, cluster_names, None)

    # raw specific
    with pytest.raises(ValueError, match="'labels' should be"):
        if isinstance(inst, BaseRaw):
            Segmentation(labels, inst, cluster_centers, cluster_names, None)
        if isinstance(inst, Epochs):
            Segmentation(np.zeros((len(inst), inst.times.size)), inst,
                         cluster_centers, cluster_names, None)

    # unsupported predict_parameters
    caplog.clear()
    Segmentation(labels, inst, cluster_centers, cluster_names, dict(test=101))
    assert "key 'test' in predict_parameters" in caplog.text
