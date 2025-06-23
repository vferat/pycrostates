"""
Microstate Segmentation
=======================

This tutorial introduces the concept of segmentation.
"""

# %%
# .. include:: ../../../../links.inc

# %%
# Segmentation
# ------------
#
# We start by fitting a modified K-means
# (:class:`~pycrostates.cluster.ModKMeans`) with a sample dataset. For more
# details about fitting a clustering algorithm, please refer to
# :ref:`this tutorial <sphx_glr_generated_auto_tutorials_cluster_00_modK.py>`.
#
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
#
# .. note::
#
#     This tutorial uses ``seaborn`` for enhanced visualization. Use the
#     following installation method appropriate for your environment:
#
#     - ``pip install seaborn``
#     - ``conda install -c conda-forge seaborn``

# sphinx_gallery_thumbnail_number = 2

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from mne.io import read_raw_eeglab

from pycrostates.cluster import ModKMeans
from pycrostates.datasets import lemon

raw_fname = lemon.data_path(subject_id="010017", condition="EC")
raw = read_raw_eeglab(raw_fname, preload=False)
raw.crop(0, 60)  # crop the dataset to speed up computation
raw.load_data()
raw.set_eeg_reference("average")  # Apply a common average reference

ModK = ModKMeans(n_clusters=5, random_state=42)
ModK.fit(raw, n_jobs=2)
ModK.reorder_clusters(order=[4, 1, 3, 2, 0])
ModK.rename_clusters(new_names=["A", "B", "C", "D", "F"])
ModK.plot()

# %%
# Once a set of cluster centers has been fitted, It can be used to predict the
# microstate segmentation with the method
# :meth:`pycrostates.cluster.ModKMeans.predict`. It returns either a
# `~pycrostates.segmentation.RawSegmentation` or an
# `~pycrostates.segmentation.EpochsSegmentation` depending on the object to
# segment. Below, the provided instance is a continuous `~mne.io.Raw` object:

segmentation = ModK.predict(
    raw,
    reject_by_annotation=True,
    factor=10,
    half_window_size=10,
    min_segment_length=5,
    reject_edges=True,
)

# %%
# The ``factor`` and ``half_window_size`` arguments control the smoothing
# algorithm introduced by Pascual Marqui\ :footcite:p:`Marqui1995`. This
# algorithm corrects wrong label assignments which occur during periods of low
# signal to noise ratio. For each timepoint, it takes into account the label of
# the neighboring segments to favor an assignment that increases label
# continuity (i.e. long segments). The ``factor`` parameter controls the
# strength of the influence of the neighboring segments while the
# ``half_window_size`` parameter controls how far (in time) neighboring
# segments have an influence.
#
# A second outlier rejection algorithm can be used on top. Controlled by the
# ``min_segment_length`` parameter, it re-assigns segments of short duration to
# their neighboring segments.

# %%
# The label of each datapoints are stored in the ``labels`` attribute.

segmentation.labels

# %%
# The segmentation can be visualized with the method
# :meth:`~pycrostates.segmentation.RawSegmentation.plot`.

segmentation.plot(tmin=1, tmax=5)
plt.show()

# %%
# Microstates parameters
# ----------------------
#
# The usual parameters used in microstate analysis can be computed with the
# method
# :meth:`~pycrostates.segmentation.RawSegmentation.compute_parameters`. This
# method returns a `dict` with a key corresponding to each combination of
# ``cluster_center name`` and ``parameter name``.

parameters = segmentation.compute_parameters()
parameters

# %%
# Global Explained Variance
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``gev`` is the total explained variance expressed by a given
# state. It is computed as the sum of global explained variance values of each
# time point assigned to a given state.

x = ModK.cluster_names
y = [parameters[elt + "_gev"] for elt in x]

ax = sns.barplot(x=x, y=y)
ax.set_xlabel("Microstates")
ax.set_ylabel("Global explained Variance (ratio)")
plt.show()

# %%
# Mean correlation
# ~~~~~~~~~~~~~~~~
#
# ``mean_corr`` corresponds to the mean correlation value of each time point
# assigned to a given state.

x = ModK.cluster_names
y = [parameters[elt + "_mean_corr"] for elt in x]

ax = sns.barplot(x=x, y=y)
ax.set_xlabel("Microstates")
ax.set_ylabel("Mean correlation")
plt.show()

# %%
# Time coverage
# ~~~~~~~~~~~~~
#
# ``timecov`` corresponds to the proportion of time during which a given state
# is active.

x = ModK.cluster_names
y = [parameters[elt + "_timecov"] for elt in x]

ax = sns.barplot(x=x, y=y)
ax.set_xlabel("Microstates")
ax.set_ylabel("Time Coverage (ratio)")
plt.show()

# %%
# Mean durations
# ~~~~~~~~~~~~~~
#
# ``meandurs`` corresponds to the mean temporal duration of segments assigned
# to a given state.

x = ModK.cluster_names
y = [parameters[elt + "_meandurs"] for elt in x]

ax = sns.barplot(x=x, y=y)
ax.set_xlabel("Microstates")
ax.set_ylabel("Mean duration (s)")
plt.show()

# %%
# Occurrence per second
# ~~~~~~~~~~~~~~~~~~~~~
#
# ``occurrences`` indicates the mean number of segments assigned to a given
# state per second. This metrics is expressed in segment per second.

x = ModK.cluster_names
y = [parameters[elt + "_occurrences"] for elt in x]

ax = sns.barplot(x=x, y=y)
ax.set_xlabel("Microstates")
ax.set_ylabel("Occurrences (segment/s)")
plt.show()

# %%
# Distributions
# ~~~~~~~~~~~~~
#
# By setting ``return_dist=True`` the underlying distributions used to computed
# those metrics are returned.

parameters = segmentation.compute_parameters(return_dist=True)
parameters

# %%
# For example, the distribution of ``C`` segment durations can be plotted.

sns.displot(parameters["C_dist_durs"], stat="probability", bins=30)
plt.show()

# %%
# Or it can be used to compute other custom metrics from the segmentation. For
# instance, the median.

median = np.median(parameters["C_dist_durs"])
print(f"Microstate C segments have a median duration of {median:.2f}s.")

# %%
# Transition probabilities
# ------------------------
#
# The obsered transition probabilities (``T``) can be retrieved with the method
# :meth:`~pycrostates.segmentation.RawSegmentation.compute_transition_matrix`.

T_observed = segmentation.compute_transition_matrix()

# %%
# This method returns a `~numpy.array` of shape ``(n_clusters, n_clusters)``
# containing the value corresponding to the chosen statistic:
#
# - ``count`` will return the number of transitions observed.
# - ``probability`` and ``proportion`` will both return the normalized
#   transition probability.
# - ``percent`` will return the normalized transition probability as
#   a percentage.
#
# The rows of ``T`` correspond to the ``"from"`` state while the columns
# correspond to the ``"to"`` state.
#
# .. note::
#
#     `~numpy.array` are 0-indexed, thus ``T[1, 2]`` encodes the transition
#     from the second to the third state (``B`` to ``C`` in this tutorial).
#
# .. note::
#
#     The transition ``from`` and ``to``  unlabeled segments are ignored and
#     not taken into account during normalization.

ax = sns.heatmap(
    T_observed,
    annot=True,
    cmap="Blues",
    xticklabels=segmentation.cluster_names,
    yticklabels=segmentation.cluster_names,
)
ax.set_ylabel("From")
ax.set_xlabel("To")
plt.show()

# %%
# It is however important to take into account the time coverage of each
# microstate in order to interpret this matrix. The more a state is
# present, the more there is a chance that a transition towards this state is
# observed without reflecting any particular dynamics in state transitions.
#
# ``pycrostates`` has the possibility to generate a theoretical transition
# matrix for a given segmentation with the method
# :meth:`~pycrostates.segmentation.RawSegmentation.compute_expected_transition_matrix`.
# This transition matrix is based on segments counts of each state present in
# the segmentation but ignores the temporal dynamics (effectively randomizing
# the transition order).

T_expected = segmentation.compute_expected_transition_matrix()
ax = sns.heatmap(
    T_expected,
    annot=True,
    cmap="Blues",
    xticklabels=segmentation.cluster_names,
    yticklabels=segmentation.cluster_names,
)
ax.set_ylabel("From")
ax.set_xlabel("To")
plt.show()

# %%
# The difference between the observed transition probability matrix
# ``T_observed`` and the theoretical transition probability matrix
# ``T_expected`` reveals particular dynamic present in the segmentation.
# Here we observe that the transition from state ``B`` to state ``A`` appears
# in larger proportion than expected while the transition from state ``B`` to
# state ``C`` is less observed than expected.

ax = sns.heatmap(
    T_observed - T_expected,
    annot=True,
    cmap="RdBu_r",
    vmin=-0.15,
    vmax=0.15,
    xticklabels=segmentation.cluster_names,
    yticklabels=segmentation.cluster_names,
)
ax.set_ylabel("From")
ax.set_xlabel("To")
plt.show()

# %%
# References
# ----------
# .. footbibliography::
