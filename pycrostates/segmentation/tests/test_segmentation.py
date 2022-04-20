from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.datasets import testing

from pycrostates.cluster import ModKMeans


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
def test_RawSegmentation_properties():
    segmentation = ModK_raw.predict(raw)
    assert isinstance(segmentation._raw, mne.io.BaseRaw)
    assert isinstance(segmentation._cluster_centers_, np.ndarray)
    assert isinstance(segmentation._cluster_names, list)
    assert isinstance(segmentation._labels, np.ndarray)
    assert isinstance(segmentation.predict_parameters, dict)


def test_RawSegmentation_plot_cluster_centers():
    segmentation = ModK_raw.predict(raw)
    segmentation.plot_cluster_centers()
    plt.close('all')


def test_RawSegmentation_plot():
    segmentation = ModK_raw.predict(raw)
    segmentation.plot()
    plt.close('all')
    segmentation.plot(tmin=0, tmax=5)
    plt.close('all')
    segmentation.plot(cmap='plasma')
    plt.close('all')
    f, ax = plt.subplots(1, 1)
    segmentation.plot(axes=ax)
    plt.close('all')
    f, ax = plt.subplots(1, 1)
    segmentation.plot(cbar_axes=ax)
    plt.close('all')


def test_RawSegmentation_compute_parameters():
    segmentation = ModK_raw.predict(raw)
    assert segmentation.__repr__()
    assert segmentation._repr_html_()  # test _repr_html_
    d = segmentation.compute_parameters(norm_gfp=False)
    assert isinstance(d, dict)
    assert '1_dist_corr' not in d.keys()


def test_RawSegmentation_compute_metrics_norm_gfp():
    segmentation = ModK_raw.predict(raw)
    assert segmentation._labels.ndim == 1
    d = segmentation.compute_parameters(norm_gfp=True)
    assert isinstance(d, dict)


def test_RawSegmentation_compute_metrics_return_dist():
    segmentation = ModK_raw.predict(raw)
    d = segmentation.compute_parameters(norm_gfp=False, return_dist=True)
    assert isinstance(d, dict)
    assert '1_dist_corr' in d.keys()


def test_EpochsSegmentation_compute_parameters():
    segmentation = ModK_epochs.predict(epochs)
    assert segmentation.__repr__()
    assert segmentation._repr_html_()  # test _repr_html_
    d = segmentation.compute_parameters(norm_gfp=False)
    assert isinstance(d, dict)


def test_EpochsSegmentation_properties():
    segmentation = ModK_raw.predict(epochs)
    assert isinstance(segmentation._epochs, mne.Epochs)
    assert isinstance(segmentation._cluster_centers_, np.ndarray)
    assert isinstance(segmentation._cluster_names, list)
    assert isinstance(segmentation._labels, np.ndarray)
    assert segmentation._labels.ndim == 2
    assert isinstance(segmentation.predict_parameters, dict)


def test_EpochsSegmentation_plot():
    segmentation = ModK_raw.predict(epochs)
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
