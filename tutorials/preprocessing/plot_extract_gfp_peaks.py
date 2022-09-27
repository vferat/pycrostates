"""
Global Field Power peaks
========================

This example demonstrates how to extract Global Field Power (:term:`GFP`) peaks
from an EEG recording.
"""

#%%
# .. Links
#
# .. _`mne installers`: https://mne.tools/stable/install/installers.html

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
# Global Field Power (:term:`GFP`) is computed as the standard deviation of the
# sensors at a given timepoint. Local maxima of the Global Field Power
# (:term:`GFP`) are known to represent the portions of EEG data
# with highest signal-to-noise ratio\ :footcite:p:`KOENIG20161104`.
# We can use the :func:`~pycrostates.preprocessing.extract_gfp_peaks`
# function to extract samples with the highest Global Field Power.
# The minimum distance between consecutive peaks can be defined with the
# ``min_peak_distance`` argument.

from pycrostates.preprocessing import extract_gfp_peaks
gfp_data = extract_gfp_peaks(raw, min_peak_distance=3)
gfp_data

#%%
# :term:`GFP` peaks can also be extracted from an :class:`~mne.Epochs` object.

epochs = mne.make_fixed_length_epochs(raw, duration=2, preload=True)
gfp_data = extract_gfp_peaks(epochs, min_peak_distance=3)
gfp_data

#%%
# The extracted :term:`GFP` peaks can be used in other preprocessing steps,
# such as a resampling using :func:`~pycrostates.preprocessing.resample`:

from pycrostates.preprocessing import resample
# extract 1 resample of 100 random high GFP samples
resample = resample(gfp_data, n_resamples=1, n_samples=100)
resample[0]

#%%
# Or they can be used to fit a clustering algorithm on the portion of EEG data
# with the highest signal-to-noise ratio.

from pycrostates.cluster import ModKMeans
ModK = ModKMeans(n_clusters=5, random_state=42)
ModK.fit(gfp_data, n_jobs=5, verbose="WARNING")
ModK.plot()

#%%
# References
# ----------
# .. footbibliography::
