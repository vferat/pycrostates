from matplotlib import pyplot as plt
from mne import create_info
import numpy as np
import pytest

from pycrostates.io import ChInfo
from pycrostates.viz import plot_cluster_centers


def test_plot_cluster_centers():
    """Test topographic plots for cluster_centers."""
    cluster_centers = np.array([[1.1, 1, 1.2], [0.4, 0.8, 0.7]])
    info = create_info(['Oz', 'Cz', 'Fpz'], sfreq=1, ch_types='eeg')
    info.set_montage('standard_1020')
    chinfo = ChInfo(info)

    # plot with info
    plot_cluster_centers(cluster_centers, info)
    plt.close('all')

    # plot with chinfo
    plot_cluster_centers(cluster_centers, chinfo)
    plt.close('all')

    # provide cluster_names
    plot_cluster_centers(cluster_centers, info, ['A', 'B'])
    plt.close('all')

    # provide ax
    f, ax = plt.subplots(1, 2)
    plot_cluster_centers(cluster_centers, chinfo, axes=ax)
    plt.close('all')
    f, ax = plt.subplots(2, 1)
    plot_cluster_centers(cluster_centers, info, axes=ax)
    plt.close('all')

    # provide show
    plot_cluster_centers(cluster_centers, info, show=True)
    plt.close('all')
    plot_cluster_centers(cluster_centers, info, show=False)
    plt.close('all')

    # invalid arguments
    cluster_centers_ = [[1.1, 1, 1.2], [0.4, 0.8, 0.7]]
    with pytest.raises(TypeError, match="'cluster_centers' must be an "):
        plot_cluster_centers(cluster_centers_, info)
    with pytest.raises(TypeError, match="'info' must be an instance of"):
        plot_cluster_centers(cluster_centers, info=101)
    with pytest.raises(TypeError, match="'cluster_names' must be an "):
        plot_cluster_centers(cluster_centers, info=chinfo, cluster_names=101)
    with pytest.raises(TypeError, match="'cluster_names' must be an "):
        plot_cluster_centers(cluster_centers, info=chinfo, cluster_names=101)
    with pytest.raises(TypeError, match="'axes' must be an "):
        plot_cluster_centers(cluster_centers, info=chinfo, axes=101)
    with pytest.raises(TypeError, match="'block' must be an "):
        plot_cluster_centers(cluster_centers, info=chinfo, block=101)
    with pytest.raises(TypeError, match="'block' must be an "):
        plot_cluster_centers(cluster_centers, info=chinfo, block=0)

    # info without montage
    with pytest.raises(RuntimeError, match="No digitization points found"):
        info_ = create_info(['Cpz', 'Cz', 'Fpz'], sfreq=1, ch_types='eeg')
        plot_cluster_centers(cluster_centers, info_)

    # mismatch
    with pytest.raises(ValueError,
                       match="Argument 'cluster_centers' and 'cluster_names'"):
        plot_cluster_centers(cluster_centers, info=chinfo, cluster_names=['A'])
    f, ax = plt.subplots(1, 1)
    with pytest.raises(ValueError,
                       match="Argument 'cluster_centers' and 'axes' must "):
        plot_cluster_centers(cluster_centers, info=chinfo, axes=ax)
    plt.close('all')

    # invalid show
    with pytest.raises(TypeError, match="'show' must be an "):
        plot_cluster_centers(cluster_centers, info, show=101)
