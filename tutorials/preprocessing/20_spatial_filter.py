"""
Spatial Filter
==============

This example demonstrates the effect of spatial filter on EEG data.
"""

#%%
# .. include:: ../../../../links.inc

#%%
# .. note::
#
#     The lemon datasets used in this tutorial is composed of EEGLAB files. To
#     use the MNE reader :func:`mne.io.read_raw_eeglab`, the ``pymatreader``
#     optional dependency is required. Use the following installation method
#     appropriate for your environment:
#
#     - ``pip install pymatreader``
#     - ``conda install -c conda-forge pymatreader``
#
#     Note that an environment created via the `MNE installers`_ includes
#     ``pymatreader`` by default.

import matplotlib.pyplot as plt
import numpy as np
from mne.io import read_raw_eeglab
from mne.channels import find_ch_adjacency
from mne.viz import plot_ch_adjacency, plot_topomap

from pycrostates.datasets import lemon
from pycrostates.preprocessing import apply_spatial_filter

raw_fname = lemon.data_path(subject_id='010004', condition='EC')
raw = read_raw_eeglab(raw_fname, preload=True)

raw.pick('eeg')
raw.set_eeg_reference('average')

#%%
# Spatial filters were first introduced in the context of EEG source imaging
# in\ :footcite:t:`michel2019eeg`, with the aim to reduce the impact of
# local transient artifacts on EEG channels. The computation of a spatial
# filter starts with the creation of an adjacency matrix, defining whether
# pairs of electrodes are adjacent (neighbors).
# An adjacency matrix can be represented as a graph where each node represents
# a sensor and each edge links adjacent sensors.

adjacency, ch_names = find_ch_adjacency(info=raw.info, ch_type="eeg")
plot_ch_adjacency(raw.info, adjacency, ch_names, kind="2d", edit=False)
plt.show()

#%%
# The spatial filter averages the signal locally by using the samples of each
# channel and its nearest neighbors. The maximum and minimum value of the
# neighboring channels is removed to improve the signal to noise ratio.
#
# .. note::
#
#     You can provide your own adjacency matrix using the ``adjacency`` parameter
#     of :func:`pycrostates.preprocessing.apply_spatial_filter`. To create a custom
#     matrix you can either edit the default one by setting the ``edit`` parameter
#     to ``True`` in :func:`mne.viz.plot_ch_adjacency` or create a
#     :class:`scipy.sparse.csr_matrix` or :class:`numpy.ndarray` from scratch.

raw_filter = raw.copy()
apply_spatial_filter(raw_filter, n_jobs=-1)

#%%
# To assess the impact of the spatial filter, we can display topographies from
# random points in the recording.

n = 10
random_sample = sorted(np.random.randint(0, raw.n_times, n))
fig, axes = plt.subplots(nrows=2, ncols=n, figsize=(15, 4))
for s, sample in enumerate(random_sample):
    plot_topomap(
        raw.get_data()[:, sample],
        pos=raw.info,
        axes=axes[0, s],
        sphere=np.array([0,0,0,0.1]),
        show=False,
    )
    plot_topomap(
        raw_filter.get_data()[:, sample],
        pos=raw_filter.info,
        axes=axes[1, s],
        sphere=np.array([0,0,0,0.1]),
        show=False,
    )
    axes[0, s].set_title(f"Sample\n{sample}")
axes[0, 0].set_ylabel("Raw")
axes[1, 0].set_ylabel("Filtered")
fig.tight_layout()
plt.show()

# After applying the spatial filter, the topographies are smoother.
# These topographies, presenting less local artifact, can be used as input for
# a clustering algorithm.

#%%
# References
# ----------
# .. footbibliography::
