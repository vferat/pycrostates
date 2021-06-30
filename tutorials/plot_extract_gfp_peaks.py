"""
Global field power peaks extraction
===========================

This example demonstrates how to extract global field power (gfp) peaks.
"""

from mne.io import read_raw_edf
from mne.datasets import eegbci
from mne.channels import make_standard_montage
from pycrostates.clustering import ModKMeans

subject = 1
runs = [1]

raw_fnames = eegbci.load_data(subject, runs, update_path=True)[0]
raw = read_raw_edf(raw_fnames, preload=True)
eegbci.standardize(raw)  # set channel names

raw.rename_channels(lambda x: x.strip('.'))
montage = make_standard_montage('standard_1005')
raw.set_montage(montage)

raw.pick('eeg')
raw.set_eeg_reference('average')
# %%
# Gfp peaks extraction can be done in two ways: the first one using the :func:`~pycrostates.preprocessing.extract_gfp_peaks`.
# Note that this function also works for :class:`mne.epochs.Epochs` but will alwas return a :class:`mne.io.Raw` obejct.
from pycrostates.preprocessing import extract_gfp_peaks
gfp_peaks = extract_gfp_peaks(raw, min_peak_distance=3)
gfp_peaks
# %%
# then fit the gfp peaks to the modified Kmeans algorithm. Make sure not to change the default parameter min_peak_distance=0
# to avoid performing GFP extraction on your already extracted gfp peaks.
n_clusters = 4
ModK = ModKMeans(n_clusters=n_clusters, random_state=42)
ModK.fit(gfp_peaks, n_jobs=5, min_peak_distance=0)
ModK.plot()

# %%
# or directly extract gfp peaks during the fitting using the min_peak_distance parameter of :meth:`ModK.fit`.
n_clusters = 4
ModK = ModKMeans(n_clusters=n_clusters, random_state=42)
ModK.fit(raw, min_peak_distance=3, n_jobs=5)
ModK.plot()