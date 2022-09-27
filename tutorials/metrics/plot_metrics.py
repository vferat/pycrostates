"""
Evaluating clustering fits
==========================

Since microstate analysis is based on an empirical model, it is impossible to
know the ground truth label of each sample. It is therefore necessary to rely
on reliable metrics in order to evaluate the relevance of the parameters used
in our model, and more particularly the number of :term:`cluster centers`.
"""

#%%
# .. Links
#
# .. _`mne installers`: https://mne.tools/stable/install/installers.html

# %%
# Pycrostates implements several metrics to evaluate the quality of
# clustering solutions without knowing the ground truth:
#
# * Silhouette score\ :footcite:p:`Silhouettes`:
#   :func:`~pycrostates.metrics.silhouette_score` (higher the better)
# * Calinski-Harabasz score\ :footcite:p:`Calinski-Harabasz`:
#   :func:`~pycrostates.metrics.calinski_harabasz_score` (higher the better)
# * Dunn score\ :footcite:p:`Dunn`:
#   :func:`~pycrostates.metrics.dunn_score` (higher the better)
# * Davies-Bouldin score\ :footcite:p:`Davies-Bouldin`:
#   :func:`~pycrostates.metrics.davies_bouldin_score` (lower the better)
#
# Those metrics can directly be applied to a fitted clustering algorithm
# such as the :py:class:`~pycrostates.cluster.ModKMeans`.
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
# In this example, we will use a single subject EEG recording in order to give
# more explanation about each of this metrics before showing an application
# case to determine the optimal number of :term:`cluster centers`
# ``n_clusters`` while fitting a :class:`~pycrostates.cluster.ModKMeans`
# algorithm.
# For this example, we start by computing each of the score on the clustering
# results of a :class:`~pycrostates.cluster.ModKMeans` fitted for different
# ``n_clusters`` ranging from 2 to 8.

from pycrostates.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    dunn_score,
    davies_bouldin_score,
)

cluster_numbers = range(2, 9)
scores = dict(silhouette=[], calinski_harabasaz=[], dunn=[], davies_bouldin=[])
for n_clusters in cluster_numbers:
    # fit K-means algorithm with a set number of cluster centers
    ModK = ModKMeans(n_clusters=n_clusters, random_state=42)
    ModK.fit(raw, n_jobs=2)

    # compute scores
    scores["silhouette"].append(silhouette_score(ModK))
    scores["calinski_harabasaz"].append(calinski_harabasz_score(ModK))
    scores["dunn"].append(dunn_score(ModK))
    scores["davies_bouldin"].append(davies_bouldin_score(ModK))

#%%
# The Silhouette score
# --------------------
# The Silhouette score\ :footcite:p:`Silhouettes` focuses on 2 metrics: the
# intra-cluster distance and the inter-cluster distance. It summarizes how well
# clusters are dense and well separated.
#
# The silhouette score is bounded between ``-1`` (low cluster separation)
# to ``1`` (high cluster separation).

plt.figure()
plt.scatter(cluster_numbers, scores["silhouette"])
plt.xlabel('n_clusters')
plt.ylabel('Silhouette score')
plt.show()

#%%
# In this example, we can observe that a number of ``n_clusters = 3``
# gives the highest score compared to other solutions thus indicating
# a better cluster separation and high cluster density.
# Note than solutions for ``n_clusters = 2`` and ``n_clusters = 4`` centers
# give score in the same order of magnitude.

# %%
# The Calinski-Harabasz score
# ---------------------------
# The Calinski-Harabasz score\ :footcite:p:`Calinski-Harabasz` is the ratio of
# the sum of inter-clusters dispersion and of intra-cluster dispersion for all
# clusters (where the dispersion is defined as the sum of correlations
# squared). As the silhouette score, it also summarizes how well clusters are
# dense and well separated. Higher values indicates higher cluster density and
# better separation.

plt.scatter(cluster_numbers, scores["calinski_harabasaz"])
plt.xlabel('n_clusters')
plt.ylabel('Calinski Harabasz score')
plt.show()

#%%
# In this example, we can observe that a number of ``n_clusters = 4``
# gives the highest score compared to other solutions thus indicating
# a better cluster separation and higher cluster density.

#%%
# The Dunn score
# --------------
#
# The Dunn score\ :footcite:p:`Dunn` is defined as a ratio of the smallest
# inter-cluster distance to the largest intra-cluster distance. Overall, it
# summarizes how well clusters are farther apart and less dispersed. Higher
# values indicates a better separation.

plt.figure()
plt.scatter(cluster_numbers, scores["dunn"])
plt.xlabel('n_clusters')
plt.ylabel('Dunn score')
plt.show()

#%%
# In this example, we can observe that a number of ``n_clusters = 8``
# gives the highest score compared to other solutions thus indicating
# a better cluster separation.

#%%
# The Davies-Bouldin score
# ------------------------
# The Davies-Bouldin score\ :footcite:p:`Davies-Bouldin` is defined as the
# average similarity measure of each cluster with its most similar cluster,
# where similarity is the ratio of intra-cluster distances to inter-cluster
# distances. Overall, it summarizes how well clusters are farther apart and
# less dispersed. Lower values indicates a better separation, ``0`` being the
# lowest score possible.

plt.figure()
plt.scatter(cluster_numbers, scores["davies_bouldin"])
plt.xlabel('n_clusters')
plt.ylabel('Davies Bouldin score')
plt.show()

#%%
# Conclusion
# ----------
# There is no global consensus on which ``n_cluster`` value to
# choose. The Silhoutette score seems to be in favor of
# ``n_clusters=3``, Calinski-Harabasz score ``n_clusters=4``,
# Dunn score ``n_clusters=8`` and Davies-Bouldin score ``n_clusters=3 or 4``.
# Each of these scores provides similar but different information about the
# quality of the clustering.
#
# Note that for microstate analysis the choice of the number of
# :term:`cluster centers` is also
# a trade off between the quality of the clustering,
# interpretations that can be made of it,
# and the variance expressed by the current clustering.
# This is why there is no perfect solution and score
# and why it is up to the person who conducts the analysis
# to evaluate the most judicious choice,
# by exploring for example several clustering solutions.

#%%
# References
# ----------
# .. footbibliography::
