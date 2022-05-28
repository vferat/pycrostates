"""
Global field power peaks extraction
===================================

This example demonstrates how to extract global field power (gfp) peaks for an eeg recording.
"""

#%%
# We start by loading some example data:

import mne
from mne.io import read_raw_eeglab

from pycrostates.datasets import lemon

raw_fname = lemon.data_path(subject_id='010004', condition='EC')
raw = read_raw_eeglab(raw_fname, preload=True)

raw.pick('eeg')
raw.set_eeg_reference('average')

#%%
# We can then use the :func:`~pycrostates.preprocessing.extract_gfp_peaks`
# function to extract samples with highest global field power.
# The min_peak_distance allow to select the minimum number of sample between 2
# selected peaks.

from pycrostates.preprocessing import extract_gfp_peaks
gfp_data = extract_gfp_peaks(raw, min_peak_distance=3)
gfp_data

#%%
# This function can also be used on :func:`~mne.epochs.Epochs`

epochs = mne.make_fixed_length_epochs(raw, duration=2, preload=True)
gfp_data = extract_gfp_peaks(epochs, min_peak_distance=3)
gfp_data

#%%
# gfp_data can the be used for otherp reprocessing steps such as :func:`~pycrostates.preprocessing.resample`

from pycrostates.preprocessing import resample
resample = resample(gfp_data, n_resamples=1, n_samples=100) #extract 1 resample of  100 random high gfp samples.
resample[0]

#%%
# Or to directly fit a clustering algorithm

from pycrostates.cluster import ModKMeans
n_clusters = 5
ModK = ModKMeans(n_clusters=n_clusters, random_state=42)
ModK.fit(gfp_data, n_jobs=5)
ModK.plot()
