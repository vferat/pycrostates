"""
The ModKmeans object
====================

This tutorial introduces the :class:`pycrostates.clustering.ModKMeans` structure in detail.
"""

from mne.io import read_raw_eeglab
from sklearn import preprocessing

import mne
import pycrostates
from pycrostates.datasets import lemon
from pycrostates.clustering import ModKMeans

subject_ids = ['010020', '010021', '010022', '010023', '010024', '010026']
conditions = ['EO', 'EC']

dataframe = list()
for condition in conditions:
    for subject_id in subject_ids:
        d = dict()
        raw_fname = lemon.load_data(subject_id=subject_id, condition=condition)
        raw = read_raw_eeglab(raw_fname, preload=True)
        raw = lemon.standardize(raw)
        d['raw'] = raw
        d['condition'] = condition
        d['subject_id'] = subject_id
        dataframe.append(d)

# %%
# 
n_clusters = 5
for data in dataframe:
    ModK = ModKMeans(n_clusters=n_clusters, random_seed=42)
    gfp_peaks = pycrostates.preprocessing.extract_gfp_peaks(data['raw'])
    data['gfp_peaks'] = gfp_peaks
    ModK.fit(gfp_peaks, n_jobs=5)
    data['ModK'] = ModK

# %%
# 
group_ModK = list()
for condition in conditions:
    d = dict()
    d['condition'] = condition
    individual_topographies = list()
    for data in dataframe:
        if data['condition'] == condition:
            individual_topographies.append(data['ModK'].get_cluster_centers_as_raw())

    concat_individual_topographies = mne.concatenate_raws(individual_topographies)
    ModK = ModKMeans(n_clusters=n_clusters, random_seed=42)
    ModK.fit(concat_individual_topographies, n_jobs=5)
    d['ModK'] = ModK

    group_ModK.append(d)
# %%
# 
group_ModK[0]['ModK'].plot()
# %%
# 
group_ModK[1]['ModK'].plot()