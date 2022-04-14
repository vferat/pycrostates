from matplotlib import pyplot as plt
import mne
import numpy as np

from pycrostates.viz import plot_raw_segmentation, plot_epoch_segmentation


folder = mne.datasets.sample.data_path() / 'MEG' / 'sample'
fname = folder / 'sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw(fname, preload=False)
raw.pick_types(eeg=True).crop(tmin=0, tmax=10, include_tmax=True)
raw.load_data()
events = mne.make_fixed_length_events(raw, duration=1)
epochs = mne.Epochs(raw, events, tmin=0, tmax=0.5, baseline=None, preload=True)


def test_plot_raw_segmentation():
    """Test topographic plots for cluster_centers."""
    n_clusters = 4
    labels = np.random.choice([-1, 0, 1, 2, 3], raw.times.size)

    plot_raw_segmentation(labels=labels, raw=raw, n_clusters=n_clusters)
    plt.close('all')

    # provide ax
    f, ax = plt.subplots(1, 1)
    plot_raw_segmentation(
            labels=labels,
            raw=raw,
            n_clusters=n_clusters,
            cluster_names=None,
            tmin=None,
            tmax=None,
            cmap=None,
            axes=ax,
            cbar_axes=None,
            block=False
            )
    plt.close('all')

    # provide cbar_ax
    f, cbar_ax = plt.subplots(1, 1)
    plot_raw_segmentation(
            labels=labels,
            raw=raw,
            n_clusters=n_clusters,
            cluster_names=None,
            tmin=None,
            tmax=None,
            cmap=None,
            axes=None,
            cbar_axes=cbar_ax,
            block=False
            )
    plt.close('all')

    # provide ax and cbar_ax
    f, axes = plt.subplots(1, 2)
    plot_raw_segmentation(
            labels=labels,
            raw=raw,
            n_clusters=n_clusters,
            cluster_names=None,
            tmin=None,
            tmax=None,
            cmap=None,
            axes=axes[0],
            cbar_axes=axes[1],
            block=False
            )
    plt.close('all')

    # provide cmap
    f, axes = plt.subplots(1, 2)
    plot_raw_segmentation(
            labels=labels,
            raw=raw,
            n_clusters=n_clusters,
            cluster_names=None,
            tmin=None,
            tmax=None,
            cmap='plasma',
            axes=None,
            cbar_axes=None,
            block=False
            )
    plt.close('all')


def test_plot_epoch_segmentation():
    """Test topographic plots for cluster_centers."""
    n_clusters = 4
    labels = np.random.choice(
        [-1, 0, 1, 2, 3], (len(epochs), epochs.times.size))

    plot_epoch_segmentation(
            labels=labels,
            epochs=epochs,
            n_clusters=n_clusters,
            cluster_names=None,
            cmap=None,
            axes=None,
            cbar_axes=None,
            block=False
            )
    plt.close('all')

    # provide ax
    f, ax = plt.subplots(1, 1)
    plot_epoch_segmentation(
            labels=labels,
            epochs=epochs,
            n_clusters=n_clusters,
            cluster_names=None,
            cmap=None,
            axes=ax,
            cbar_axes=None,
            block=False
            )
    plt.close('all')

    # provide cbar_ax
    f, cbar_ax = plt.subplots(1, 1)
    plot_epoch_segmentation(
            labels=labels,
            epochs=epochs,
            n_clusters=n_clusters,
            cluster_names=None,
            cmap=None,
            axes=None,
            cbar_axes=cbar_ax,
            block=False
            )
    plt.close('all')

    # provide ax and cbar_ax
    f, axes = plt.subplots(1, 2)
    plot_epoch_segmentation(
            labels=labels,
            epochs=epochs,
            n_clusters=n_clusters,
            cluster_names=None,
            cmap=None,
            axes=axes[0],
            cbar_axes=axes[1],
            block=False
            )
    plt.close('all')

    # provide cmap
    plot_epoch_segmentation(
            labels=labels,
            epochs=epochs,
            n_clusters=n_clusters,
            cluster_names=None,
            cmap='plasma',
            axes=None,
            cbar_axes=None,
            block=False
            )
    plt.close('all')
