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
# We first start by importing some EEG data

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
# In this example, we will used the
# [Silhouette Coefficient](https://doi.org/10.1016/0377-0427(87)90125-7) to
# evaluate the
# Pycrostates implementations relies on sklearn implementation
# while ensuring that the metric used for distance computations is
# the opposite value of the absolute spatial correlation.
# More details can be found on the [sklearn User Guide](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score)

from pycrostates.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    dunn_score,
    davies_bouldin_score,
)

cluster_numbers = range(2,12)
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
