"""
Group level analysis from individual gfp peaks
====================

In this tutorial, we will learn how to conduct group level analysis
by computing group level topogrpahies based on individual gfp peaks.
"""

#%%
# We first start by importing some EEG data.

from mne.io import read_raw_eeglab

from pycrostates.cluster import ModKMeans
from pycrostates.datasets import lemon
from pycrostates.preprocessing import extract_gfp_peaks, resample
from pycrostates.io import ChData

condition = 'EO'
subject_ids = ['010020', '010021' , '010022', '010023', '010024']

#%%
# In this example, we first extract individual
# GFP peaks. Then we concatenate them into
# a single dataset in order to submit it to
# clustering (group level analysis).

import numpy as np

ModK = ModKMeans(n_clusters=5,  random_state=42)
n_jobs = 2

individual_gfp_peaks = list()
for subject_id in subject_ids:
    #Load Data
    raw_fname = lemon.data_path(subject_id=subject_id, condition=condition)
    raw = read_raw_eeglab(raw_fname, preload=True)
    raw = lemon.standardize(raw)
    raw.pick('eeg')
    # For sake of time, we only use 30s of recording.
    raw.crop(0, 30)
    raw.set_eeg_reference('average')
    # Extract GFP peaks
    gfp_peaks = extract_gfp_peaks(raw)
    # Equalize peak number accross subjects
    gfp_peaks = resample(gfp_peaks, n_resamples=1, n_samples=880, random_state=42)[0]
    individual_gfp_peaks.append(gfp_peaks.get_data())
    
individual_gfp_peaks = np.hstack(individual_gfp_peaks)
individual_gfp_peaks = ChData(individual_gfp_peaks, raw.info)
# Group level clustering
ModK.fit(individual_gfp_peaks, n_jobs=n_jobs)
ModK.plot();

#%%
# We can reorganize our clustering results to our needs.

ModK.reorder_clusters(order=[0, 2, 4, 3, 1])
ModK.rename_clusters(new_names=['A', 'B', 'C', 'D', 'F'])
ModK.plot();

#%%
# We can now backfit the group level maps
# to each individual recording and extract
# microstate parameters.

import pandas as pd

ms_data = list()
for subject_id in subject_ids:
    #Load Data
    raw_fname = lemon.data_path(subject_id=subject_id, condition=condition)
    raw = read_raw_eeglab(raw_fname, preload=True)
    raw = lemon.standardize(raw)
    raw.pick('eeg')
    raw.crop(0, 30) # for sake of time
    raw.set_eeg_reference('average')
    segmentation = ModK.predict(raw, factor=10, half_window_size=8)
    d = segmentation.compute_parameters()
    d['subject_id'] = subject_id
    ms_data.append(d)

ms_data = pd.DataFrame(ms_data)
ms_data
