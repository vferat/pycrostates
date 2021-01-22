"""
Single Subject Segmentation
===========================

This example demonstrates how to segment a single subject recording into microstates sequence.
"""

import os.path as op
import mne
from mne.datasets import sample

import pycrostates
from pycrostates.clustering import ModKMeans

# %%
# This is the first section!
# The `#%%` signifies to Sphinx-Gallery that this text should be rendered as
# rST and if using one of the above IDE/plugin's, also signifies the start of a
# 'code block'.

data_path = sample.data_path()
fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'

evoked = mne.read_evokeds(fname_evoked, condition=0, baseline=(None, 0))
evoked.pick('eeg')
evoked.interpolate_bads()
evoked.set_eeg_reference('average')
# %%
# Fit.

n_clusters = 4
ModK = ModKMeans(n_clusters=n_clusters)
ModK.fit(evoked)

# %%
# Plot.

ModK.plot_cluster_centers()

# %%
# Predict.
segmentation = ModK.predict(evoked, half_window_size=5, factor=30)
pycrostates.viz.plot_segmentation(segmentation, evoked)
