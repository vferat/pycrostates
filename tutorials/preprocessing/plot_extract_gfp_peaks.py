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

raw_fname = lemon.load_data(subject_id="010004", condition="EC")
raw = read_raw_eeglab(raw_fname, preload=True)

raw.pick("eeg")
raw.set_eeg_reference("average")

#%%
# We can then use the :func:`~pycrostates.preprocessing.extract_gfp_peaks`
# function to extract samples with highest global field power.
# The min_peak_distance allow to select the minimum number of sample beween 2
# selected peaks.

from pycrostates.preprocessing import extract_gfp_peaks

raw_peaks = extract_gfp_peaks(raw, min_peak_distance=3)
raw_peaks

#%%
#
# .. warning::
#
#    The returned object will always be a :class:`~mne.io.Raw`, but should not
#    be used for any other purpose than fitting a clustering algorithm. To
#    avoid any misuse of this object, we have deliberately assigned its
#    sampling rate to -1.

raw_peaks.info["sfreq"]

#%%
# Note that this function can also be used on :func:`~mne.epochs.Epochs` but
# will always return a :class:`~mne.io.Raw` instance.

epochs = mne.make_fixed_length_epochs(raw, duration=2, preload=True)
epochs_peaks = extract_gfp_peaks(epochs, min_peak_distance=3)
epochs_peaks
