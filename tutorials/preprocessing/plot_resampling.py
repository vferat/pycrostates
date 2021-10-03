"""
Resampling
==========

This example demonstrates how resemple a recording.
"""

# %%
# We start by loading some example data:
import mne
from mne.io import read_raw_edf
from mne.datasets import eegbci
from mne.channels import make_standard_montage
subject = 1
runs = [1]

raw_fnames = eegbci.load_data(subject, runs, update_path=True)[0]
raw = read_raw_edf(raw_fnames, preload=True)

# %%
# We can now use the :func:`~pycrostates.preprocessing.resample`function to draw n_epochs of n_samples
# for our recording where n_sample defines the number of sample cointained in each epoch
# and n_epochs defines the number of epochs to draw.
from pycrostates.preprocessing import resample
resamples = resample(raw, n_epochs=10, n_samples=150, random_state=40)
resamples
# %%
# We can also use the 'coverage' parameter to automatically compute one of the two preivous parameters
# based on the amount of original data we want to cover.
# for exemple by setting n_epochs and coverage:
resamples = resample(raw, n_epochs=10, coverage=0.5, random_state=40)

# %% or by setting n_samples and coverage:
resamples = resample(raw, n_samples=150, coverage=0.5, random_state=40)

# %%
# Finally, we can also use this function to resample :func:`~mne.epochs.Epochs`
epochs = mne.make_fixed_length_epochs(raw, duration=2, preload=True)
resamples = resample(epochs, n_samples=150, coverage=0.5, random_state=40)