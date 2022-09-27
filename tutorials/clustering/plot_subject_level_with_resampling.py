"""
Subject level analysis with resampling
======================================

This tutorial introduces how to use :class:`pycrostates.preprocessing.resample`
to compute individual microstate topographies and study the stability of
the clustering results.
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

from mne.io import read_raw_eeglab

from pycrostates.cluster import ModKMeans
from pycrostates.datasets import lemon


raw_fname = lemon.data_path(subject_id='010017', condition='EC')
raw = read_raw_eeglab(raw_fname, preload=True)
raw.crop(0, 30)

raw.pick('eeg')
raw.set_eeg_reference('average')

#%%
# Resampling can be useful to study the stability and reliability of clustering results.
# In this example, we will split our data in ``n_resamples`` resamples each containing
# ``n_samples`` randomly selected from the original recording.
#
# .. note::
#
#      This method can also be used on GFP peaks only !
#      """
#      gfp_peaks = pycrostates.preprocessing.extract_gfp_peaks(raw)
#      resamples = resample(
#      raw, n_resamples=10, n_samples=1000, random_state=42
#      )
#      """
from pycrostates.preprocessing import resample

resamples = resample(
    raw, n_resamples=10, n_samples=1000, random_state=42
)

#%% We can compute the :term:`cluster centers` on each of the resample:

resample_results = []
for resample in resamples:
    ModK = ModKMeans(n_clusters=5, random_state=42)
    ModK.fit(resample, n_jobs=2)
    ModK.plot()
    resample_results.append(ModK.cluster_centers_)


#%%
# As we can see, each resampling solution explain about 70% of the resample.
# However, topographies can vary a lot from one solution to another.
# The sparsity of results reveals how data sampling may influence
# the results of microstates analysis.
# To tackle this issue, one solution could be to compute a second round
# of clustering on the concatenated resample solutions.
import numpy as np
from pycrostates.io import ChData

all_resampling_results = np.vstack(resample_results).T
all_resampling_results = ChData(all_resampling_results, ModK.info)

ModK = ModKMeans(n_clusters=5, random_state=42)
ModK.fit(all_resampling_results, n_jobs=2)
ModK.plot()

# .. note::
#      This method can also be used for group level analysis by mixing individual resampling results
#      together.
