"""
The ModKmeans object
===========================

This tutorial introduces the :class:`pycrostates.clustering.ModKMeans` structure in detail.
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
# The modified Kmeans can be instanciated with the number of cluster centers n_clusters to compute.
# By default, the modified Kmeans will only work with EEG data, but this can be modified thanks to the ''picks'' parameter.
# A random_state can be defined during class definition in order to have reproducible results.
n_clusters = 4
ModK = ModKMeans(n_clusters=n_clusters, random_state=42)

# %%
# Most methods need the modified Kmeans to be fitted. This can be done with either :class:`mne.io.Raw`: or :class:`mne.epochs.Epcohs`: data structures:
# Global field power peaks extraction can be perform on the fly by setting the min_peak_distance parameter != 0. If min_peak_distance parameter = 0 no extraction is done
# and the data is used as it is.
# Note that, depending on your setup, you can change ``n_jobs=1`` in order to use parallel processing and reduce computation time.
ModK.fit(raw, min_peak_distance=2, n_jobs=5)

# %%
# Now that our algorithm is fitted, we can visualise the cluster centers, also called microstate maps or microstate topographies
# using :meth:`ModK.plot`. Note than this method uses the :class:`~mne.Info` object of the fitted instance to display
# the topographies.
ModK.plot()

# %%
# One can access the cluster centers as a numpy array thanks to :meth:`ModK.get_cluster_centers`:
ModK.get_cluster_centers()

# %%
# or as a :class:`mne.io.Raw` object:
ModK.get_cluster_centers_as_raw()

# %%
# Clusters centers can be reordered using :meth:`ModK.reorder`:
ModK.reorder([3,2,0,1])
ModK.plot()

# %%
# and renamed using :meth:`ModK.rename`:
ModK.rename_clusters(['A', 'B', 'C', 'D'])
ModK.plot()

# %%
# Maps polarities can be inverted thanks to :meth:`ModK.invert_polarity` method. Note that this only affects visualisation:
# this has not effect during backfitting as polarities are ignored.
ModK.invert_polarity([False, False, True, True])
ModK.plot()

# %%
# Finaly, the modified Kmeans can be used to predict the microstates segmentation using the :meth:`ModK.predict` method:
# By default, semgents annoated as bad will no be labeled: this behavior can be changed by changing the `reject_by_annotation` paramerts.
# Smoothing can be performed on the output sequence by setting the `factor` parameter > 0 (no smotthing by default factor = 0) while the
# `half_window_size` parameter is used to specify the smoothing temporal span.
# Finaly, the `rejected_first_last_segments` parameter allows not to assign the first and last segment of each record (or each epoch) as these can be incomplete.
# Should have little impact for raw, but can be important when working with epochs.

segmentation = ModK.predict(raw, reject_by_annotation=True, factor=10, half_window_size=30, rejected_first_last_segments=True)
segmentation.plot()