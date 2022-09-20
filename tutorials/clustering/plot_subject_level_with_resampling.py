"""
Subject level analysis with resampling
======================================

This tutorial introduces how to use :class:`pycrostates.preprocessing.resample`
to compute individual microstate topographies and study the stability of
the clustering results.
"""

#%%
# .. note::
#
#     The lemon datasets is composed of EEGLAB files. To use the MNE reader
#     :func:`mne.io.read_raw_eeglab`, the ``pymatreader`` optional dependency
#     is required. Use the following installation method appropriate for your
#     environment:
#
#     - ``pip install pymatreader``
#     - ``conda install -c conda-forge pymatreader``
#
#     Note that an environment created via the MNE installers includes
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
# The modified K-means can be instantiated with the number of cluster centers
# ``n_clusters`` to compute. By default, the modified K-means will only work
# with EEG data, but this can be modified thanks to the ``picks`` parameter.
# A ``random_state`` can be defined during class definition in order to have
# reproducible results.
from pycrostates.preprocessing import resample

resamples = resample(
    raw, n_resamples=10, n_samples=1000, random_state=42
)

resample_results = []
for resample in resamples:
    ModK = ModKMeans(n_clusters=5, random_state=42)
    ModK.fit(resample, n_jobs=2)
    ModK.plot()
    resample_results.append(ModK.cluster_centers_)
    

#%%
# We can reorganize our clustering results to our needs.
import numpy as np
from pycrostates.io import ChData

all_resampling_results = np.vstack(resample_results).T
all_resampling_results = ChData(all_resampling_results, ModK.info)

ModK = ModKMeans(n_clusters=5, random_state=42)
ModK.fit(all_resampling_results, n_jobs=2)
ModK.plot()
