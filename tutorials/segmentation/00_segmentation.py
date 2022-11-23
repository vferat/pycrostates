"""
Microstate Segmentation
=======================

This tutorial introduces the concept of segmentation.
"""

#%%
# .. include:: ../../../../links.inc

#%%
# Here we fit a modified K-means (:class:`~pycrostates.cluster.ModKMeans` with
# a sample dataset. For more details about the fitting procedure, please refer
# to :ref:`this tutorial
# <sphx_glr_generated_auto_tutorials_cluster_00_modK.py>`.
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

#%%
# Once a set of cluster centers has been fitted, we can use it to predict the
# microstate segmentation using the
# :meth:`pycrostates.cluster.ModKMeans.predict` method. It returns either a
# `~pycrostates.segmentation.RawSegmentation` or an
# `~pycrostates.segmentation.EpochsSegmentation` depending on the provided
# instance. Below, the provided instance is a continuous `~mne.io.Raw` object:

segmentation = ModK.predict(
    raw,
    reject_by_annotation=True,
    factor=10,
    half_window_size=10,
    min_segment_length=5,
    reject_edges=True,
)

#%%
# The ``factor`` and ``half_window_size`` arguments control the smoothing
# algorithm introduced by Pascual Marqui \ :footcite:p:`Marqui1995`. This
# algorithm corrects wrong label assignment which occur during periods of low
# signal to noise ratio. For each timepoint, it takes into account the label of
# the neighbouring segments to favor assignments that increase label
# continuity (i.e. long segments). The ``factor`` parameter controls the
# strength of the influence of the neighboring segments while the
# ``half_window_size`` parameter controls how far (in time) neighboring
# segments have an influence.
#
# A second outlier rejection algorithm can be used on top. Controlled by the
# ``min_segment_length`` parameter, it re-assigns of short duration to their
# neighboring segments.

#%%
# Labels of each datapoints are stored in the ``labels`` attribute.

segmentation.labels

#%%
# The segmentation can be visualized with the method
# :meth:`~pycrostates.segmentation.RawSegmentation.plot`.

segmentation.plot(tmin=1, tmax=5)

#%%
# The usual parameters used in microstate analysis can be computed with the
# method
# :meth:`~pycrostates.segmentation.RawSegmentation.compute_parameters`.

parameters = segmentation.compute_parameters()
parameters

#%%
# This method returns a `dict` which keys corresponds to each
# combination of ``cluster_center_name`` and parameter name.
# For example, ``gev`` is the total explained variance expressed by a given
# state. It is computed as the sum of global explained variance values of each
# time point assigned to a given state.

x = ModK.cluster_names
y = [parameters['A_gev'],
     parameters['B_gev'],
     parameters['C_gev'],
     parameters['D_gev'],
     parameters['F_gev']]

ax = sns.barplot(x=x, y=y)
ax.set_xlabel('Microstates')
ax.set_ylabel('Global explained Variance (ratio)')

#%%
# ``mean_corr`` indicates the mean correlation value of each time point
# assigned to a given state.

x = ModK.cluster_names
y = [parameters['A_mean_corr'],
     parameters['B_mean_corr'],
     parameters['C_mean_corr'],
     parameters['D_mean_corr'],
     parameters['F_mean_corr']]

ax = sns.barplot(x=x, y=y)
ax.set_xlabel('Microstates')
ax.set_ylabel('Mean correlation')

#%%
# ``timecov`` indicates the proportion of time during which a given state is
# active.

x = ModK.cluster_names
y = [parameters['A_timecov'],
     parameters['B_timecov'],
     parameters['C_timecov'],
     parameters['D_timecov'],
     parameters['F_timecov']]

ax = sns.barplot(x=x, y=y)
ax.set_xlabel('Microstates')
ax.set_ylabel('Time Coverage (ratio)')

#%%
# ``meandurs`` indicates the mean temporal duration of segments assigned to a
# given state.

x = ModK.cluster_names
y = [parameters['A_meandurs'],
     parameters['B_meandurs'],
     parameters['C_meandurs'],
     parameters['D_meandurs'],
     parameters['F_meandurs']]

ax = sns.barplot(x=x, y=y)
ax.set_xlabel('Microstates')
ax.set_ylabel('Mean duration (s)')

#%%
# ``occurrences`` indicates the mean number of segments assigned to a given
# state per second. This metrics is expressed in segment per second ( . / s).
x = ModK.cluster_names
y = [parameters['A_occurrences'],
     parameters['B_occurrences'],
     parameters['C_occurrences'],
     parameters['D_occurrences'],
     parameters['F_occurrences']]

ax = sns.barplot(x=x, y=y)
ax.set_xlabel('Microstates')
ax.set_ylabel('Occurences (segment/s)')

#%%
# By setting the ``return_dist`` parameter to ``True``, one can also have
# access to the underlying distributions used to compute previous metrics.

parameters = segmentation.compute_parameters(return_dist=True)
parameters

#%%
# For example, distribution of segment durations can be plotted using
# ``seaborn``.

sns.displot(parameters['C_dist_durs'], stat='probability', bins=30)

#%%
# This can also be used to derivates other metrics form the segmentation.

median = np.median(parameters['C_dist_durs'])
mean = np.mean(parameters['C_dist_durs'])
print(
    f"Microstate C segments have a median duration of {median:.2f}s "
    f"and a mean duration of {mean:.2f}s."
)

#%%
# Finally, one can get the observed transition probabilities using the method
# :meth:`~pycrostates.segmentation.RawSegmentation.compute_transition_matrix`.

T_observed = segmentation.compute_transition_matrix()

#%%
# This method returns a `~numpy.array` of shape ``(n_clusters, n_clusters)``
# containing the value corresponding to the chosen statistic:
#
# - ``count`` will return the number of transitions observed.
# - ``probability`` and ``proportion`` will both return the normalize
#   transition probability.
# - ``percent`` will return the probability as percentage
#   (``probability`` * 100).
#
# The rows of T correspond to the "from" state while the columns indicate the
# "to" state, so that T[1,2] encodes the transition from the 2nd state to the
# 3rd state (B -> C in this example).
#
# .. note::
#
#     Transition ``from`` and ``to``  unlabeled segments are ignored and not
#     taken into account during normalization.

ax = sns.heatmap(T_observed,
                 annot=True,
                 cmap="Blues",
                 xticklabels=segmentation.cluster_names,
                 yticklabels=segmentation.cluster_names)
ax.set_ylabel("From")
ax.set_xlabel("To")

#%%
# It is however important to take into account the time coverage of each
# microstate in order to analyze this matrix. Indeed, the more a state is
# present, the more there is a chance that a transition towards this state is
# observed without reflecting any particular dynamics in state transitions.
# Pycrostates offers the possibility to generate a theoretical transition
# matrix thanks to the method
# :meth:`~pycrostates.segmentation.RawSegmentation.compute_expected_transition_matrix`.
# This transition matrix is based on segments counts of each state present in
# the segmentation but ignores the temporal dynamics of segmentation (random
# transition order).

T_expected = segmentation.compute_expected_transition_matrix()
ax = sns.heatmap(T_expected,
                 annot=True,
                 cmap="Blues",
                 xticklabels=segmentation.cluster_names,
                 yticklabels=segmentation.cluster_names)
ax.set_ylabel("From")
ax.set_xlabel("To")

#%%
# The difference between the observed transition probability matrix
# ``T_observed`` and the theoretical transition probability matrix
# ``T_expected`` reveals particular dynamic present in the segmentation.
# Here we observe that the transition from state B to state C appears
# in much larger proportion than expected while the transition from
# state B to state C is much less observed than expected.
ax = sns.heatmap(T_observed - T_expected,
                 annot=True,
                 cmap="RdBu_r",
                 vmin=-0.15,
                 vmax=0.15,
                 xticklabels=segmentation.cluster_names,
                 yticklabels=segmentation.cluster_names)
ax.set_ylabel("From")
ax.set_xlabel("To")

#%%
# References
# ----------
# .. footbibliography::
