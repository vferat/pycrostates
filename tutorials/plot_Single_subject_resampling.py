"""
Single Subject Segmentation with resampling
===========================

This example demonstrates how to segment a single subject recording into microstates sequence.
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
# This step is optional. We can extract GfP peaks before doing resampling.
from pycrostates.preprocessing import extract_gfp_peaks
raw = extract_gfp_peaks(raw, min_peak_distance=3)
raw

# %%
# Resample raw instance into 10 epochs of 150 samples
from pycrostates.preprocessing import resample
resamples = resample(raw, n_epochs=10, n_samples=150, random_state=40)
# %%
# Compute Kmeans clustering on each sample independently
n_clusters = 4
ModK = ModKMeans(n_clusters=n_clusters, random_state=42)

resample_centers = list()
for sample in resamples:
    ModK.fit(sample, n_jobs=5)
    cluster_centers = ModK.get_cluster_centers_as_raw()
    resample_centers.append(cluster_centers)
    

# %%
# Then compute Kmeans clustering on the concatenated results
from mne import concatenate_raws
concat_raw = concatenate_raws(resample_centers)
ModK.fit(concat_raw, n_jobs=5)
ModK.plot()