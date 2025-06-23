"""
Resampling
==========

This example demonstrates how to resample a recording.
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

import mne
from mne.io import read_raw_eeglab

from pycrostates.datasets import lemon


raw_fname = lemon.data_path(subject_id="010004", condition="EC")
raw = read_raw_eeglab(raw_fname, preload=True)
raw.crop(0, 30)

raw.pick("eeg")
raw.set_eeg_reference("average")

# %%
# The :func:`~pycrostates.preprocessing.resample` function
# draws ``n_resamples`` of ``n_samples`` from a recording where ``n_samples``
# defines the number of samples contained in each epoch and ``n_resamples``
# defines the number of epochs to draw.

from pycrostates.preprocessing import resample

resamples = resample(raw, n_resamples=3, n_samples=150, random_state=40)
resamples

# %%
# The ``coverage`` parameter can be used to automatically compute one of
# the two previous parameters based on the amount of original data to
# cover. For example, by setting ``n_resamples`` and ``coverage``:

resamples = resample(raw, n_resamples=3, coverage=0.5, random_state=40)
resamples

# %%
# or by setting ``n_samples`` and ``coverage``:

resamples = resample(raw, n_samples=1000, coverage=0.5, random_state=40)
resamples

# %%
# Finally, this function can be used to resample :class:`~mne.Epochs`.

epochs = mne.make_fixed_length_epochs(raw, duration=2, preload=True)
resamples = resample(epochs, n_samples=150, coverage=0.5, random_state=40)
resamples
