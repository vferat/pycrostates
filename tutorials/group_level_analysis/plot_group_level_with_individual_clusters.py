"""
Group-level analysis from individual clusters
=============================================

In this tutorial, we will learn how to conduct group level analysis
by computing group-level topographies based on individual clusters.
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

# sphinx_gallery_thumbnail_number = 2

import numpy as np
from mne.io import read_raw_eeglab

from pycrostates.cluster import ModKMeans
from pycrostates.datasets import lemon
from pycrostates.io import ChData
from pycrostates.preprocessing import extract_gfp_peaks


condition = "EO"
subject_ids = ["010020", "010021", "010022", "010023", "010024"]

#%%
# In this example, we start with a subject level analysis by computing
# individual topographies from :term:`GFP` peaks. Then, the individual
# topographies are concatenated and used in the group level analysis to fit a
# second clustering algorithm.

individual_cluster_centers = list()
for subject_id in subject_ids:
    # load Data
    raw_fname = lemon.data_path(subject_id=subject_id, condition=condition)
    raw = read_raw_eeglab(raw_fname, preload=True)
    raw = lemon.standardize(raw)
    raw.pick("eeg")
    raw.crop(0, 30)  # crop the dataset to speed up computation
    raw.set_eeg_reference("average")
    # extract GFP peaks
    gfp_peaks = extract_gfp_peaks(raw)
    # subject level clustering
    ModK = ModKMeans(n_clusters=5, random_state=42)
    ModK.fit(gfp_peaks, n_jobs=2)
    individual_cluster_centers.append(ModK.cluster_centers_)

group_cluster_centers = np.vstack(individual_cluster_centers).T
group_cluster_centers = ChData(group_cluster_centers, ModK.info)

# group level clustering
ModK = ModKMeans(n_clusters=5, random_state=42)
ModK.fit(group_cluster_centers, n_jobs=2)
ModK.plot()

#%%
# The :term:`cluster centers` can be re-organize to our needs.

ModK.reorder_clusters(order=[4, 2, 0, 1, 3])
ModK.rename_clusters(new_names=["MS1", "MS2", "MS3", "MS4", "MS5"])
ModK.plot()

#%%
# We can now backfit the group level maps to each individual recording and
# extract microstate parameters.

import pandas as pd

ms_data = list()
for subject_id in subject_ids:
    # Load Data
    raw_fname = lemon.data_path(subject_id=subject_id, condition=condition)
    raw = read_raw_eeglab(raw_fname, preload=True)
    raw = lemon.standardize(raw)
    raw.pick("eeg")
    raw.crop(0, 30)  # for sake of time
    raw.set_eeg_reference("average")
    segmentation = ModK.predict(raw, factor=10, half_window_size=8)
    d = segmentation.compute_parameters()
    d["subject_id"] = subject_id
    ms_data.append(d)

ms_data = pd.DataFrame(ms_data)
ms_data
