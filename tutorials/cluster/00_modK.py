"""
The ModKMeans object
====================

This tutorial introduces the main clustering object
:class:`pycrostates.cluster.ModKMeans` structure in detail. The Modified
K-means algorithm is based on :footcite:t:`Marqui1995`.
"""

#%%
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
# The modified K-means\ :footcite:p:`Marqui1995` can be instantiated with the
# number of :term:`cluster centers` ``n_clusters`` to fit. By default, the
# modified K-means will only work with EEG data, but other channel types can be
# selected with the ``picks`` argument.
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
#     Fitting a clustering algorithm is a computationally expensive operation.
#     Depending on your configuration, you can change the argument ``n_jobs``
#     to take advantage of multiprocessing to reduce computation time.

ModK.fit(raw, n_jobs=5)

#%%
# Now that our algorithm is fitted, we can visualize the
# :term:`cluster centers`, also called microstate maps or microstate
# topographies using :meth:`pycrostates.cluster.ModKMeans.plot`.
# Setting ``show_gradient`` will plot a line between maximum and minimum
# value of each :topography to help visualization.

ModK.plot(show_gradient=True)

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
