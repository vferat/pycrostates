"""
Resampling
==========

This example demonstrates how resemple a recording.
"""
import mne
from mne.io import read_raw_eeglab

from pycrostates.datasets import lemon


raw_fname = lemon.load_data(subject_id='010004', condition='EC')
raw = read_raw_eeglab(raw_fname, preload=True)
raw.crop(0, 30)

raw.pick('eeg')
raw.set_eeg_reference('average')

# %%
# We can now use the :func:`~pycrostates.preprocessing.resample`function to draw n_epochs of n_samples
# for our recording where n_sample defines the number of sample cointained in each epoch
# and n_epochs defines the number of epochs to draw.
from pycrostates.preprocessing import resample
resamples = resample(raw, n_epochs=10, n_samples=150, random_seed=40)
resamples
# %%
# We can also use the 'coverage' parameter to automatically compute one of the two preivous parameters
# based on the amount of original data we want to cover.
# for exemple by setting n_epochs and coverage:
resamples = resample(raw, n_epochs=10, coverage=0.5, random_seed=40)

# %% or by setting n_samples and coverage:
resamples = resample(raw, n_samples=150, coverage=0.5, random_seed=40)

# %%
# Finally, we can also use this function to resample :func:`~mne.epochs.Epochs`
epochs = mne.make_fixed_length_epochs(raw, duration=2, preload=True)
resamples = resample(epochs, n_samples=150, coverage=0.5, random_seed=40)
