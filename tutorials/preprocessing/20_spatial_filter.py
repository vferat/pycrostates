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
import mne
from mne.io import read_raw_eeglab

from pycrostates.datasets import lemon

raw_fname = lemon.data_path(subject_id='010004', condition='EC')
raw = read_raw_eeglab(raw_fname, preload=True)

raw.pick('eeg')
raw.set_eeg_reference('average')

#%%
# Spatial filter was first introduced in \ :footcite:p:`michel2019eeg`
# in the context of EEG source imaging. Since then, it as also been 
# widly applied as a preprocessing step of EEG microstate analysis.
# The goal of the preprocessing step is to reduced the effect of local transcient
# artefacts that corrupts EEG channels.
# The computation of this filter relies on the creation of an adjacency matrix:
import matplotlib.pyplot as plt 
from mne.channels import find_ch_adjacency
from mne.viz import plot_ch_adjacency

adjacency, ch_names = find_ch_adjacency(info=raw.info, ch_type="eeg")
plot_ch_adjacency(raw.info, adjacency, ch_names, kind='2d', edit=False)
plt.show()
#%%
# :term:`GFP` peaks can also be extracted from an :class:`~mne.Epochs` object.
from pycrostates.preprocessing import apply_spatial_filter
raw_filter = raw.copy()
apply_spatial_filter(raw_filter, n_jobs=-1)
#%%
# Let's have a look at the effect of the spatial filter.
# First we can vizualise the effect on topographies:
import numpy as np

n = 10
random_sample = np.random.randint(0, raw.n_times, n)
sphere = np.array([0,0,0,1])
fig, axes = plt.subplots(nrows=2, ncols=n, figsize=(10,2))
for s,sample in enumerate(random_sample):
    mne.viz.topomap._plot_topomap(raw.get_data()[:, sample], pos=raw.info, axes=axes[0, s], sphere=sphere, show=False)
    mne.viz.topomap._plot_topomap(raw_filter.get_data()[:, sample], pos=raw_filter.info, axes=axes[1, s], sphere=sphere, show=False)
plt.show()
#%%
# PSD
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18,5), sharex=True, sharey=True)

raw.compute_psd(fmin=0, fmax=30).plot(axes=axes[0], show=False)
axes[0].set_title("Filter Off")
raw_filter.compute_psd(fmin=0, fmax=30).plot(axes=axes[1], show=False)
axes[1].set_title("Filter On")
plt.show()

#%%
# Correlations
import seaborn as sns
correlation_matrix = np.corrcoef(raw.get_data(), raw_filter.get_data())
correlation_matrix = correlation_matrix[len(raw.info['ch_names']):, :len(raw.info['ch_names'])]
sns.displot(np.diag(correlation_matrix))
plt.show()

#%%
# References
# ----------
# .. footbibliography::
