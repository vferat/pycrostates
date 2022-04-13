from pathlib import Path

import matplotlib.pyplot as plt
import mne
from mne.datasets import testing

from pycrostates.cluster import ModKMeans


dir_ = Path(testing.data_path()) / 'MEG' / 'sample'
fname_raw_testing = dir_ / 'sample_audvis_trunc_raw.fif'
raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
raw = raw.pick('eeg')
raw = raw.filter(0, 40)
raw = raw.crop(0, 10)

events = mne.make_fixed_length_events(raw, 1)
epochs = mne.epochs.Epochs(raw, events, preload=True)

n_clusters = 4
ModK = ModKMeans(n_clusters=n_clusters)

def test_RawSegmentation_plot():
    ModK.fit(raw, n_jobs=1)
    segmentation = ModK.predict(raw)
    segmentation.plot()
    plt.close('all')
    segmentation.plot(tmin=0, tmax=5)
    plt.close('all')
    segmentation.plot(cmap='plasma')
    plt.close('all')
    f, ax = plt.subplots(1, 1)
    segmentation.plot(ax=ax)
    plt.close('all')
    f, ax = plt.subplots(1, 1)
    segmentation.plot(cbar_ax=ax)
    plt.close('all')

def test_RawSegmentation_compute_parameters():
    ModK.fit(raw, n_jobs=1)
    segmentation = ModK.predict(raw)
    assert (segmentation.__repr__())
    assert (segmentation._repr_html_())  # test _repr_html_
    d = segmentation.compute_parameters(norm_gfp=False)
    assert isinstance(d, dict)

def test_RawSegmentation_compute_parameters():
    ModK.fit(raw, n_jobs=1)
    segmentation = ModK.predict(raw)
    assert (segmentation.__repr__())
    assert (segmentation._repr_html_())  # test _repr_html_
    d = segmentation.compute_parameters(norm_gfp=False)
    assert isinstance(d, dict)
    
def test_RawSegmentation_compute_metrics_norm_gfp():
    ModK.fit(raw, n_jobs=1)
    segmentation = ModK.predict(raw)
    d = segmentation.compute_parameters(norm_gfp=True)
    assert isinstance(d, dict)

def test_EpochsSegmentation_compute_metrics():
    ModK.fit(epochs, n_jobs=1)
    segmentation = ModK.predict(epochs)
    assert (segmentation.__repr__())
    assert (segmentation._repr_html_())  # test _repr_html_
    d = segmentation.compute_parameters(norm_gfp=False)
    assert isinstance(d, dict)

def test_EpochsSegmentation_plot():
    ModK.fit(raw, n_jobs=1)
    segmentation = ModK.predict(epochs)
    segmentation.plot()
    plt.close('all')
    segmentation.plot(cmap='plasma')
    plt.close('all')
    f, ax = plt.subplots(1, 1)
    segmentation.plot(ax=ax)
    plt.close('all')
    f, ax = plt.subplots(1, 1)
    segmentation.plot(cbar_ax=ax)
    plt.close('all')
