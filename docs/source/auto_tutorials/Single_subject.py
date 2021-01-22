"""
Single Subject Segmentation
=========================

This example demonstrates how to segment a single subject recording into microstates sequence.
"""

import os.path as op
import mne
from mne.datasets import sample

from pycrostates.clustering import ModKMeans

# %%
# This is the first section!
# The `#%%` signifies to Sphinx-Gallery that this text should be rendered as
# rST and if using one of the above IDE/plugin's, also signifies the start of a
# 'code block'.

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = mne.io.read_raw_fif(raw_fname, preload=True)


raw = raw.pick('eeg')
raw = raw.crop(0, 30)
n_clusters = 4
ModK = ModKMeans(n_clusters=n_clusters)

ModK.fit(raw)
segmentation = ModK.predict(raw)