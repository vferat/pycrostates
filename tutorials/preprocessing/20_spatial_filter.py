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
import mne
from mne.io import read_raw_eeglab

from pycrostates.datasets import lemon

raw_fname = lemon.data_path(subject_id='010004', condition='EC')
raw = read_raw_eeglab(raw_fname, preload=True)

raw.pick('eeg')
raw.set_eeg_reference('average')

#%%
# Spatial filter were first introduced in the context of EEG source imaging in\ :footcite:t:`michel2019eeg`,
# These filters aim to reduce the impact of local transient artifacts on EEG channels.
# The computation of such filter starts with the creation of an adjacency matrix:
from mne.channels import find_ch_adjacency
from mne.viz import plot_ch_adjacency

adjacency, ch_names = find_ch_adjacency(info=raw.info, ch_type="eeg")
plot_ch_adjacency(raw.info, adjacency, ch_names, kind='2d', edit=False)
plt.show()
# In the context of EEG, an adjacency matrix indicate whether pairs of electrodes are adjacent (i.e neighbours) or not in the graph.
# The signal is then locally averaged by using the values of each channel and time point, as well as the values of its nearest neighbors.
# The maximum and minimum values are removed from this calculation in order to improve the signal to noise ratio.
#%%
# We can apply the spatial filter to a copy of the raw instance:
from pycrostates.preprocessing import apply_spatial_filter
raw_filter = raw.copy()
apply_spatial_filter(raw_filter, n_jobs=-1)

#%%
# To assess the impact of the spatial filter, we can display topographies from random points in the recording.
import numpy as np

n = 10
random_sample = np.random.randint(0, raw.n_times, n)
sphere = np.array([0,0,0,0.1])
fig, axes = plt.subplots(nrows=2, ncols=n, figsize=(10,2))
for s,sample in enumerate(random_sample):
    mne.viz.topomap._plot_topomap(raw.get_data()[:, sample], pos=raw.info, axes=axes[0, s], sphere=sphere, show=False)
    mne.viz.topomap._plot_topomap(raw_filter.get_data()[:, sample], pos=raw_filter.info, axes=axes[1, s], sphere=sphere, show=False)
plt.show()

# After applying the spatial filter, we observe smoother topographies.
# These topographies, presenting less local artfeact, can be used as input for a clstering algorithm.

#%%
# References
# ----------
# .. footbibliography::