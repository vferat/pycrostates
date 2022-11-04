"""
Microstate Segmentation
=======================

This tutorial introduces the concept of segmentation.
"""

#%%
# .. include:: ../../../../links.inc

#%%
# Here we fit a modified K-means with a sample dataset.
# For more details about the fitting procedure,
# please refer to :ref:`this tutorial <fit>`.

from mne.io import read_raw_eeglab

from pycrostates.cluster import ModKMeans
from pycrostates.datasets import lemon

raw_fname = lemon.data_path(subject_id='010017', condition='EC')
raw = read_raw_eeglab(raw_fname, preload=True)
raw.crop(0, 60)  # crop the dataset to speed up computation
raw.pick('eeg')  # select EEG channels
raw.set_eeg_reference('average')  # Apply a common average reference

n_clusters = 5
ModK = ModKMeans(n_clusters=n_clusters, random_state=42)

ModK.fit(raw, n_jobs=5)
ModK.reorder_clusters(order=[4, 1, 3, 2, 0])
ModK.rename_clusters(new_names=['A', 'B', 'C', 'D', 'F'])
ModK.plot()

#%%
# Once a set a cluster centers has been computed, we can use it  to
# predict the microstate segmentation using the :meth:`pycrostates.cluster.ModKMeans.predict`
# method. This method returns either a `~pycrostates.segmentation.RawSegmentation` or
# `~pycrostates.segmentation.EpochsSegmentation` depending on the input instance.
segmentation = ModK.predict(raw, reject_by_annotation=True, factor=10,
                            half_window_size=10, min_segment_length=5,
                            reject_edges=True)

#%%
# The ``factor`` and ``half_window_size`` allows control of the smoothing algorithm introduced
# by Pascual Marqui \ :footcite:p:`Marqui1995`. This algorithm allows to correct wrong label assignment
# which occur during periods of low signal to noise ratio. For each timepoint,
# this algorithm takes into account the label of the neighbouring segments in order to favor assignments that increase
# label continuity
# (i.e. long segments). The ``factor`` parameter controls the strength of the influence of the neighboring segments
# while the ``half_window_size`` parameter controls how far (in time) neighbours should have influence.
#
# After obtaining the segmentation (with or without smoothing) a second outlier rejection algorithm can be used.
# It is controlled by the ``min_segment_length`` parameter, which when different from zero, allows to reassign
# segments of short duration to their neighboring labels.

#%%
# Labels of each datapoints are stored in the ``_labels`` attribute

segmentation._labels

#%%
# The :meth:`pycrostates.cluster.ModKMeans.plot` method can be used
# to visualize the segmentation:

# sphinx_gallery_thumbnail_number = 2
segmentation.plot(tmin=1, tmax=5)

#%%
# Parameters usually studied in microstate analysis can be computed using the 
# :meth:`~pycrostates.segmentation.RawSegmentation.compute_parameters` method.

parameters = segmentation.compute_parameters()
parameters

#%%
# This method returns a `dict` which keys corresponds to each
# combinaison of ``cluster_center_name`` and parameter name.
# For example, ``gev`` is the total explained variance expressed by a given state.
# It is computed as the sum of global explained variance values of each time point assigned
# to a given state.
import matplotlib.pyplot as plt
import seaborn as sns
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
# ``mean_corr`` indicates the mean correlation value of each time point assigned to a given state.
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
# ``timecov`` indicates the proportion of time during which a given state is active.
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
# ``meandurs`` indicates the mean temporal duration of segments assigned to a given state
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
# ``occurrences`` indicates the mean number of segments assigned to a given state
# per second. This metrics is expressed in segment per second ( . / s).
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
# By setting the ``return_dist`` parameter to ``True``
# one can also have access to the underlying distributions
# used to compute previous metrics.

parameters = segmentation.compute_parameters(return_dist=True)
parameters

#%%
# For example, distribution of segment durations can be plotted using ``seaborn``:
sns.displot(parameters['C_dist_durs'], stat='probability', bins=30)

#%%
# This can also be used to derivates other metrics form the segmentation:
import numpy as np

median = np.median(parameters['C_dist_durs'])
mean = np.mean(parameters['C_dist_durs'])
print(f'Microstate C segment has a median duration of {median:.2f}s and mean duration of {mean:.2f}s')

#%%
# Finally, get the observed transition probabilities using the
# :meth:`~pycrostates.segmentation.RawSegmentation.compute_transition_probabilities` method.

T = segmentation.get_transition_matrix()

#%%
# This method returns a `numpy.array` of shape ``(n_clusters, n_clusters)`` containing the
# value corresponding to the chosen statistic:
#  - ``count`` will return the number of transitions observed.
#  - ``probability`` and ``proportion`` will both return the normalize transition probability.
#  - ``percent`` will return the probability as percentage ( ``probability`` * 100).
#
# The rows of T correspond to the "from" state while the columns indicate the "to" state,
# so that T[1,2] encodes the transition from the 2nd state to the 3rd state (B -> C in this example).
#
# .. note::
#
#     Transition ``from`` and ``to``  unlabeled segments are ignored and not taken into
#     account during normalization.


import matplotlib.pyplot as plt
import seaborn as sns
ax = sns.heatmap(T,
                 annot=True,
                 cmap='Blues',
                 xticklabels=segmentation.cluster_names,
                 yticklabels=segmentation.cluster_names)
ax.set_ylabel('From')
ax.set_xlabel('To')

plt.show()

#%%
# It is however important to take into account the time coverage of each microstate in order to analyze this matrix.
# Indeed, the more a state is present, the more there is a chance that a transition towards this state is observed
# without reflecting any particular dynamics in state transitions.
# Pycrostates offers the possibility to generate a theoretical transition matrix based on segments counts of each state 
# present in the segmentation but ignoring the temporal dynamics of segmentation (random order).
Te = segmentation.get_expected_transition_matrix()
ax = sns.heatmap(Te,
                 annot=True,
                 cmap='Blues',
                 xticklabels=segmentation.cluster_names,
                 yticklabels=segmentation.cluster_names)
ax.set_ylabel('From')
ax.set_xlabel('To')

plt.show()

#%%
# The difference between the observed transition probability matrix ``T``
# and the theoretical transition probability matrix ``Te``
# reveals particular dynamic present in the segmentation.
# Here we observe that the transition from state B to state C appears
# in much larger proportion than expected while the transition from 
# state B to state C is much less observed than in theory.
ax = sns.heatmap(T - Te,
                 annot=True,
                 cmap='RdBu_r',
                 vmin=-0.15,
                 vmax=0.15,
                 xticklabels=segmentation.cluster_names,
                 yticklabels=segmentation.cluster_names)
ax.set_ylabel('From')
ax.set_xlabel('To')

plt.show()
#%%
# References
# ----------
# .. footbibliography::
