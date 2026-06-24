"""
Group level analysis from individual GFP peaks
==============================================

In this tutorial, we will learn how to conduct group level analysis
by computing group level topographies based on individual
:term:`Global Field Power` (:term:`GFP`) peaks.
"""

# %%
# .. include:: ../../../../links.inc

# %%
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

from pycrostates.cluster import ModKMeans
from pycrostates.datasets import lemon
from pycrostates.io import ChData
from pycrostates.preprocessing import extract_gfp_peaks, resample


condition = "EO"
subject_ids = ["010020", "010021", "010022", "010023", "010024"]

# %%
# In this example, we start by extracting the individual :term:`GFP` peaks and
# concatenating them into a single dataset. The totality of :term:`GFP` peaks
# is used in the group level analysis to fit a clustering algorithm.

individual_gfp_peaks = list()
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
    # equalize peak number across subjects by resampling
    gfp_peaks = resample(gfp_peaks, n_resamples=1, n_samples=880, random_state=42)[0]
    individual_gfp_peaks.append(gfp_peaks.get_data())

individual_gfp_peaks = np.hstack(individual_gfp_peaks)
individual_gfp_peaks = ChData(individual_gfp_peaks, raw.info)

# group level clustering
ModK = ModKMeans(n_clusters=5, random_state=42)
ModK.fit(individual_gfp_peaks, n_jobs=2)
ModK.plot()

# %%
# The :term:`cluster centers` can be re-organize to our needs.

ModK.reorder_clusters(order=[0, 2, 4, 3, 1])
ModK.rename_clusters(new_names=["A", "B", "C", "D", "F"])
ModK.plot()

# %%
# We can now use this fitted clustering algorithm to predict the segmentation
# on each individual. This is also referred to as backfitting the group level
# maps to each individual recording. Finally, we can extract microstate
# parameters from the backfitted segmentations.

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

# %%
# From this point on, we can visualize our results and do a statistical
# analysis. For example we can plot the :term:`GEV` of each microstate class.

data = [[d["A_gev"], d["B_gev"], d["C_gev"], d["D_gev"], d["F_gev"]] for d in ms_data]

plt.violinplot(np.array(data))
plt.title("Global Explained Variance (%)")
plt.xticks(
    ticks=range(1, len(ModK.cluster_names) + 1),
    labels=ModK.cluster_names,
)
plt.show()
