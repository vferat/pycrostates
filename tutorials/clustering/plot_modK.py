"""
The ModKmeans object
====================

This tutorial introduces the :class:`pycrostates.cluster.ModKMeans`
structure in detail.
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
# The modified Kmeans can be instantiated with the number of cluster centers
# ``n_clusters`` to compute. By default, the modified K-means will only work
# with EEG data, but this can be modified thanks to the ``picks`` parameter.
# A random_state can be defined during class definition in order to have
# reproducible results.

n_clusters = 5
ModK = ModKMeans(n_clusters=n_clusters, random_state=42)

#%%
# Most methods need the modified K-means to be fitted. This can be done with
# either :class:`mne.io.Raw` or :class:`mne.epochs.Epochs` data structures.
# Note that, depending on your setup, you can change ``n_jobs=1`` in order to
# use parallel processing and reduce computation time.

ModK.fit(raw, n_jobs=5)

#%%
# Now that our algorithm is fitted, we can visualize the cluster centers, also
# called microstate maps or microstate topographies using
# :meth:`pycrostates.cluster.ModKMeans.plot`.

ModK.plot()

#%%
# One can access the cluster centers as a numpy array thanks to the
# ``cluster_centers_`` attribute:

ModK.cluster_centers_

#%%
# Clusters centers can be reordered using
# :meth:`pycrostates.cluster.ModKMeans.reorder_clusters`.

ModK.reorder_clusters(order=[3, 2, 4, 0, 1])
ModK.plot()

#%%
# and renamed using :meth:`pycrostates.cluster.ModKMeans.rename_clusters`.

ModK.rename_clusters(new_names=['A', 'B', 'C', 'D', 'F'])
ModK.plot()

#%%
# Maps polarities can be inverted using
# :meth:`pycrostates.cluster.ModKMeans.invert_polarity`.
# method. Note that it only affects visualization, it has not effect during
# backfitting as polarities are ignored.

ModK.invert_polarity([False, False, True, True, False])
ModK.plot()

#%%
# Finally, the modified K-means can be used to predict the microstates
# segmentation using the :meth:`pycrostates.cluster.ModKMeans.predict` method.
# By default, segments annotated as bad will not be labeled, but this behavior
# can be changed with the ``reject_by_annotation`` argument. Smoothing can be
# performed on the output sequence by setting the ``factor`` argument ``> 0``
# (no smoothing by default ``factor=0``) while the ``half_window_size``
# parameter is used to specify the smoothing temporal span. Finally, the
# ``reject_edges`` argument allows not to assign the first and last segment of
# each recording (or each epoch) as these can be incomplete. It should have
# little impact for raw, but can be important when working with epochs.

segmentation = ModK.predict(raw, reject_by_annotation=True, factor=10,
                            half_window_size=10, min_segment_length=5,
                            reject_edges=True)
segmentation.plot(tmin=1, tmax=5)
