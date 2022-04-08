from matplotlib import pyplot as plt
import mne
import numpy as np
import pytest

from pycrostates.viz import plot_raw_segmentation

folder = mne.datasets.sample.data_path() / 'MEG' / 'sample'
raw = mne.io.read_raw(folder / 'sample_audvis_filt-0-40_raw.fif', preload=True)
raw.pick_types(meg=False, eeg=True)
raw.crop(tmin=0, tmax=10, include_tmax=True)

def test_plot_segmentation():
    """Test topographic plots for cluster_centers."""
    data = raw.get_data()
    cluster_centers = np.random.randint(-10, 10, (data.shape[0],4))
    labels = np.random.choice([-1,0,1,2,4], data.shape[-1])

    plot_raw_segmentation(
            labels=labels,
            inst=raw,
            cluster_centers=cluster_centers,
            cluster_names=None,
            tmin=0.0,
            tmax= None,
            cmap=None,
            ax=None,
            cbar_ax=None,
            block=False
            )
    plt.close('all')

    # provide ax
    f, ax = plt.subplots(1, 1)
    plot_raw_segmentation(
            labels=labels,
            inst=raw,
            cluster_centers=cluster_centers,
            cluster_names=None,
            tmin=0.0,
            tmax= None,
            cmap=None,
            ax=ax,
            cbar_ax=None,
            block=False
            )
    plt.close('all')

    # provide cbar_ax
    f, cbar_ax = plt.subplots(1, 1)
    plot_raw_segmentation(
            labels=labels,
            inst=raw,
            cluster_centers=cluster_centers,
            cluster_names=None,
            tmin=0.0,
            tmax= None,
            cmap=None,
            ax=None,
            cbar_ax=cbar_ax,
            block=False
            )
    plt.close('all')

    # provide ax and cbar_ax
    f, axes = plt.subplots(1, 2)
    plot_raw_segmentation(
            labels=labels,
            inst=raw,
            cluster_centers=cluster_centers,
            cluster_names=None,
            tmin=0.0,
            tmax= None,
            cmap=None,
            ax=axes[0],
            cbar_ax=axes[1],
            block=False
            )
    plt.close('all')

    # provide cmap
    f, axes = plt.subplots(1, 2)
    plot_raw_segmentation(
            labels=labels,
            inst=raw,
            cluster_centers=cluster_centers,
            cluster_names=None,
            tmin=0.0,
            tmax= None,
            cmap='plasma',
            ax=None,
            cbar_ax=None,
            block=False
            )
    plt.close('all')



