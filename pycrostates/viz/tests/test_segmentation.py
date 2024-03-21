import mne
import numpy as np
from matplotlib import pyplot as plt

from pycrostates.viz import plot_epoch_segmentation, plot_raw_segmentation

folder = mne.datasets.testing.data_path() / "MEG" / "sample"
fname = folder / "sample_audvis_trunc_raw.fif"
raw = mne.io.read_raw(fname, preload=False)
raw.pick_types(eeg=True).crop(tmin=0, tmax=10, include_tmax=True)
raw.load_data().filter(1, 40)
events = mne.make_fixed_length_events(raw, duration=1)
epochs = mne.Epochs(raw, events, tmin=0, tmax=0.5, baseline=None, preload=True)


def test_plot_raw_segmentation():
    """Test segmentation plots for raw."""
    n_clusters = 4
    labels = np.random.choice([-1, 0, 1, 2, 3], raw.times.size)

    plot_raw_segmentation(labels, raw, n_clusters)

    # provide ax
    f, ax = plt.subplots(1, 1)
    plot_raw_segmentation(labels, raw, n_clusters, axes=ax)

    # provide cbar_ax
    f, cbar_ax = plt.subplots(1, 1)
    plot_raw_segmentation(labels, raw, n_clusters, cbar_axes=cbar_ax)

    # provide ax and cbar_ax
    f, axes = plt.subplots(1, 2)
    plot_raw_segmentation(labels, raw, n_clusters, axes=axes[0], cbar_axes=axes[1])

    # provide cmap
    plot_raw_segmentation(labels, raw, n_clusters, cmap="plasma")


def test_plot_epoch_segmentation():
    """Test segmentation plots for epochs."""
    n_clusters = 4
    labels = np.random.choice([-1, 0, 1, 2, 3], (len(epochs), epochs.times.size))

    plot_epoch_segmentation(labels, epochs, n_clusters)

    # provide ax
    f, ax = plt.subplots(1, 1)
    plot_epoch_segmentation(labels, epochs, n_clusters, axes=ax)

    # provide cbar_ax
    f, cbar_ax = plt.subplots(1, 1)
    plot_epoch_segmentation(labels, epochs, n_clusters, cbar_axes=cbar_ax)

    # provide ax and cbar_ax
    f, axes = plt.subplots(1, 2)
    plot_epoch_segmentation(labels, epochs, n_clusters, axes=axes[0], cbar_axes=axes[1])

    # provide cmap
    plot_epoch_segmentation(labels, epochs, n_clusters, cmap="plasma")
