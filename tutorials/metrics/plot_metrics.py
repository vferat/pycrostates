"""
Choosing the number of cluster centers k.
=========================================

In this tutorial, we will see how to get indications on the best
number of clusters centers k to use.
It is important to note that there is no single solution to this choice.
It will always be a trade off between the quality of the clustering,
the possibility to analyze the microstates in the light of the literature
and the variance expressed by the decomposition.
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

import matplotlib.pyplot as plt
from mne.io import read_raw_eeglab

from pycrostates.cluster import ModKMeans
from pycrostates.datasets import lemon

raw_fname = lemon.data_path(subject_id="010017", condition="EO")
raw = read_raw_eeglab(raw_fname, preload=True)
raw.crop(0, 30)

raw.pick("eeg")
raw.set_eeg_reference("average")

#%%
# Pycrostates implements several metrics to evaluate the quality of
# clustering solutions without knowing the ground truth:
# - The silhouette score (higher the better)
# - The Calinski Harabasz score (higher the better)
# - The Dunn score (higher the better)
# - The Davies Bouldin score (lower the better)
#
# We can apply these metrics on clustering solution
# computed with different values of n_clusters and
# analyse which give the best clustering solution.
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
    ModK.fit(raw, n_jobs=12)
    # silhouettee
    silhouette_scores.append(silhouette_score(ModK))
    # calinski and harabasz
    calinski_harabasz_scores.append(calinski_harabasz_score(ModK))
    # dunn
    dunn_scores.append(dunn_score(ModK))
    # davies bouldin
    davies_bouldin_scores.append(davies_bouldin_score(ModK))

#%%
# Silhouette score

plt.figure()
plt.scatter(cluster_numbers, silhouette_scores)
plt.xlabel('n_clusters')
plt.ylabel('Silhouette score')
plt.show()

#%%
# Calinski Harabasz score

plt.scatter(cluster_numbers, calinski_harabasz_scores)
plt.xlabel('n_clusters')
plt.ylabel('Calinski Harabasz score')
plt.show()

#%%
# Dunn score

plt.figure()
plt.scatter(cluster_numbers, dunn_scores)
plt.xlabel('n_clusters')
plt.ylabel('Dunn score')
plt.show()

#%%
# Davies Bouldin

plt.figure()
plt.scatter(cluster_numbers, davies_bouldin_scores)
plt.xlabel('n_clusters')
plt.ylabel('Davies Bouldin score')
plt.show()

#%%
# As can be seen, there is no global consensus on which n_cluster value to choose.
# The silhoutette score seems to be in favor of
# ``n_clusters=3``, Calinski Harabasz score for ``n_clusters=4``,
# Dunns core for ``n_clusters=8`` and Davies Bouldin score for ``n_clusters= 3 or 4``.
# It is then up to the scientist to choose which compromise
# he/she wants to make for this analysis
