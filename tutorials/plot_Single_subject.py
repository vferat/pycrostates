"""
Single Subject Segmentation
===========================

This example demonstrates how to segment a single subject recording into microstates sequence.
"""

from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.channels import make_standard_montage
import pandas as pd

import pycrostates
from pycrostates.clustering import ModKMeans
from pycrostates.metrics import compute_metrics

subject = 1
runs = [1,2,3]  # motor imagery: hands vs feet

raw_fnames = eegbci.load_data(subject, runs, update_path=True)
raw_files = [read_raw_edf(f, preload=True) for f in raw_fnames]
raw = concatenate_raws(raw_files)
eegbci.standardize(raw)  # set channel names

raw.rename_channels(lambda x: x.strip('.'))
montage = make_standard_montage('standard_1005')
raw.set_montage(montage)

raw.pick('eeg')
raw.set_eeg_reference('average')
# %%
# Fit the modified Kmeans algorithm with the raw data. Here we use ``gfp=True`` to extract gfp peaks on fly.
# Note that, depending on your setup, you can change ``n_jobs=1`` in order to use parallel processing and speed up the process.

n_clusters = 4
ModK = ModKMeans(n_clusters=n_clusters)
ModK.fit(raw, gfp=True, n_jobs=5)

# %%
# Now that our algorithm is fitted, we can visualise the cluster centers, also called Microstate maps or Microstate topographies
# using :meth:`ModK.plot_cluster_centers`. Note than this method uses the :class:`~mne.Info` object of the fitted instance to display
# the topographies.
ModK.plot_cluster_centers()

# %%
# Predict.
segmentation = ModK.predict(raw, half_window_size=5, factor=10)
pycrostates.viz.plot_segmentation(segmentation, raw)

# %%
# Compute microstate parameters and convert results into a :class:`~pandas.DataFrame`.
metrics = compute_metrics(raw, ModK.cluster_centers, norm_gfp=True,  half_window_size=5, factor=10)
df = pd.DataFrame([metrics])
print(df)