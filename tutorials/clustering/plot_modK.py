"""
The ModKMeans object
====================

This tutorial introduces the main clustering object
:class:`pycrostates.cluster.ModKMeans` structure in detail.
"""

#%%
# .. Links
#
# .. _`mne installers`: https://mne.tools/stable/install/installers.html

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
#     Note that an environment created via the `MNE installers`_. includes
#     ``pymatreader`` by default.

from mne.io import read_raw_eeglab

from pycrostates.cluster import ModKMeans
from pycrostates.datasets import lemon


# load sample dataset
raw_fname = lemon.data_path(subject_id='010017', condition='EC')
raw = read_raw_eeglab(raw_fname, preload=True)
raw.crop(0, 10)  # crop the dataset to speed up computation
raw.pick('eeg')  # select EEG channels
raw.set_eeg_reference('average')  # Apply a common average reference

#%%
# The modified K-means can be instantiated with the number of
# :term:`cluster centers` ``n_clusters`` to fit. By default, the modified
# K-means will only work with EEG data, but other channel types can be selected
# with the ``picks`` argument.
#
# .. note::
#
#     A K-means algorithm starts with a random initialization. Thus, 2 separate
#     fit will not yield the same :term:`cluster centers`. To achieve
#     reproductible fits, the ``random_state`` argument can be used.

n_clusters = 5
ModK = ModKMeans(n_clusters=n_clusters, random_state=42)

#%%
# After creating a :class:`~pycrostates.cluster.ModKMeans`, the next step is to
# fit the model. In other words, fitting a clustering algorithm will determine
# the microstate maps, also called :term:`cluster centers`. A clustering
# algorithm can be fitted with :class:`~mne.io.Raw`,
# :class:`~mne.epochs.Epochs` or :class:`~pycrostates.io.ChData` objects.
#
# .. note::
#
#     Fitting a clustering algorithm is a computationaly expensive operation.
#     Depending on your configuration, you can change the argument ``n_jobs``
#     to take advantage of multiprocessing to reduce computation time.

ModK.fit(raw, n_jobs=5)

#%%
# Now that our algorithm is fitted, we can visualize the
# :term:`cluster centers`, also called microstate maps or microstate
# topographies using :meth:`pycrostates.cluster.ModKMeans.plot`.

ModK.plot()

#%%
# The :term:`cluster centers` can be retrieved as a numpy array with the
# ``cluster_centers_`` attribute.

ModK.cluster_centers_

#%%
# By default, the :term:`cluster centers` are named from ``0`` to
# ``n_clusters - 1`` and are ordered based on the fit. You can reorder
# (:meth:`pycrostates.cluster.ModKMeans.reorder_clusters`) and
# rename (:meth:`pycrostates.cluster.ModKMeans.rename_clusters`) each
# microstates to match your preference.

ModK.reorder_clusters(order=[3, 0, 1, 2, 4])
ModK.rename_clusters(new_names=['A', 'B', 'C', 'D', 'F'])
ModK.plot()

#%%
# The map polarities can be inverted using the
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
# ``reject_edges`` argument is used to prevent the assignment of the first and
# last segment of each recording (or each epoch) as these can be incomplete. It
# should have little impact for `~mne.io.Raw` objects, but can be important
# when working with `~mne.Epochs`.

segmentation = ModK.predict(raw, reject_by_annotation=True, factor=10,
                            half_window_size=10, min_segment_length=5,
                            reject_edges=True)
segmentation.plot(tmin=1, tmax=5)
