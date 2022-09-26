"""
Evaluating clustering results.
==============================

Since microstate analysis is based on an empirical model,
it is impossible to know the ground truth label of each sample.
It is therefore necessary to rely on reliable metrics in order to evaluate
the relevance of the parameters used in our model, and more particularly
the number of clusters center.
"""

# %%
# Pycrostates implements several metrics to evaluate the quality of
# clustering solutions without knowing the ground truth:
# * The silhouette score :func:`~pycrostates.metrics.silouhette.silhouette_score`(higher the better)
# * The Calinski Harabasz score :func:`~pycrostates.metrics.calinski_harabasz.calinski_harabasz_score`(higher the better)
# * The Dunn score :func:`~pycrostates.metrics.dunn.dunn_score`(higher the better)
# * The Davies Bouldin score func:`~pycrostates.metrics.davies_bouldin.davies_bouldin_score`(lower the better)
#
# Those metrics can directly be applied to a fitted clustering algortihm
# such as the :py:class:`~pycrostates.cluster.ModKmeans`.
#
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


import matplotlib.pyplot as plt
from mne.io import read_raw_eeglab

from pycrostates.cluster import ModKMeans
from pycrostates.datasets import lemon

raw_fname = lemon.data_path(subject_id="010017", condition="EO")
raw = read_raw_eeglab(raw_fname, preload=True)
raw.crop(0, 30)

raw.pick("eeg")
raw.set_eeg_reference("average")

# %%
# In this exemple , we will use a single subject EEG recording in order to
# study more give more explanation about each of this metrics
# before showing an application case to determine the optimal number of cluster
# (i.e microstate topographies) ``n_clusters`` while performing clustering with
# the :class:`~pycrostates.cluster.ModKmeans` algorithm.
# For computation cost reasons, we start by computing
# each of the score on the clustering results of
# a :class:`~pycrostates.cluster.ModKMeans` fitted for
# different values of ``n_clusters`.`

from pycrostates.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    dunn_score,
    davies_bouldin_score,
)

cluster_numbers = range(2,10)
silhouette_scores = []
calinski_harabasz_scores = []
dunn_scores = []
davies_bouldin_scores = []

for k in cluster_numbers:
    ModK = ModKMeans(n_clusters=k, random_state=42)
    ModK.fit(raw, n_jobs=2)
    # silhouettee
    silhouette_scores.append(silhouette_score(ModK))
    # calinski and harabasz
    calinski_harabasz_scores.append(calinski_harabasz_score(ModK))
    # dunn
    dunn_scores.append(dunn_score(ModK))
    # davies bouldin
    davies_bouldin_scores.append(davies_bouldin_score(ModK))

#%%
# The silhouette score
# --------------------
# Silhouette analysis compromises two scores:
# - The intra cluster distance
# - The between cluster distance
# Overall, it summurizes how well clusters are dense and well separated.
# The silhouette score is bounded between ``-1`` for poor clustering
# to ``1`` for well separated clusters.

plt.figure()
plt.scatter(cluster_numbers, silhouette_scores)
plt.xlabel('n_clusters')
plt.ylabel('Silhouette score')
plt.show()

# In this example, we can observe that a number of ``n_clusters = 3``
# gives an highest score compared to other solutions thuse indicating
# a better cluster separation and high cluster density.
# Note than solutions for ``n_clusters = 2``
# and ``n_clusters = 4`` centers give score in the same order of magnitude.

# %%
# The Calinski Harabasz score
# ---------------------------
# The index is the ratio of the sum of between-clusters dispersion
# and of within-cluster dispersion for all clusters (where dispersion
# is defined as the sum of correlations squared):
# As the silouhette index score, it also summurizes how well clusters
# are dense and well separated.
# Higher values indicates higher cluster density and better separation.

plt.scatter(cluster_numbers, calinski_harabasz_scores)
plt.xlabel('n_clusters')
plt.ylabel('Calinski Harabasz score')
plt.show()

# In this example, we can observe that a number of ``n_clusters = 4``
# gives an highest score compared to other solutions thuse indicating
# a better cluster separation and high cluster density.
#%%
# The Dunn score
# ------------------------
# The Dunn score or Dunn index defined as
# a ratio of the smallest inter-cluster distance to the largest intra-cluster distance.
# Overall, it summurizes how well clusters are farther apart and less dispersed.
# Higher Dunn score relates to a model with better clustering.

plt.figure()
plt.scatter(cluster_numbers, dunn_scores)
plt.xlabel('n_clusters')
plt.ylabel('Dunn score')
plt.show()

# In this example, we can observe that a number of `n_clusters = 8`
# gives an highest score compared to other solutions thuse indicating
# a better clustering
#%%
# The Davies-Bouldin score
# ------------------------
# Davies-Bouldin is defined as the average similarity measure of
# each cluster with its most similar cluster,
# where similarity is the ratio of within-cluster distances
# to between-cluster distances.
# Overall, it summurizes how well clusters are farther apart and less dispersed.
# Lower Davies-Bouldin index relates to a model with better
# separation between the clusters, ``0`` being the lowest score possible.

plt.figure()
plt.scatter(cluster_numbers, davies_bouldin_scores)
plt.xlabel('n_clusters')
plt.ylabel('Davies Bouldin score')
plt.show()

#%%
# Conclusion
# ----------
# As can be seen, there is no global consensus on which ``n_cluster`` value to choose.
# The silhoutette score seems to be in favor of
# ``n_clusters=3``, Calinski Harabasz score for ``n_clusters=4``,
# Dunns score for ``n_clusters=8`` and Davies Bouldin score for ``n_clusters= 3 or 4``.
# Each of these scores provides similar but different information
# about the quality of the clustering.
#
# Note that for microstate analysis the choice of the number
# of cluster centers is also
# a trade off between the quality of the clustering,
# interpretations that can be made of it,
# and the variance expressed by the current clustering.
# This is why there is no perfect solution and score
# and why it is up to the person who conducts the analysis
# to evaluate the most judicious choice,
# by exploring for example several clustering solutions.
