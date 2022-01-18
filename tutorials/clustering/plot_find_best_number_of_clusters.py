"""
Clustering performance evaluation
=================================

This example shows how to evaluate the quality of the clustering and gives an indication of how many clusters to choose.
This step can be done at different stages of the analysis, at the subject or group level for example.
Pycrostates implements different scores to evaluate the quality of clustering, we will see how to use them.
"""

from mne.io import read_raw_eeglab

from pycrostates.datasets import lemon
from pycrostates.clustering import ModKMeans


raw_fname = lemon.load_data(subject_id='010004', condition='EC')
raw = read_raw_eeglab(raw_fname, preload=True)
raw.crop(0,30)

raw.pick('eeg')
raw.set_eeg_reference('average')

# %%
# We must first fit our clustering algorithm (in our case the modified Kmeans) with some data.
n_clusters = 4
ModK = ModKMeans(n_clusters=n_clusters, random_seed=42)
ModK.fit(raw, n_jobs=5)
ModK.plot()
# %%
# We can then compute several clustering performance score on the fitted instance.
from pycrostates.metrics import silhouette, davies_bouldin, calinski_harabasz, dunn

silhouette_score = silhouette(ModK)
print('silhouette score: ', silhouette_score)
davies_bouldin_score = davies_bouldin(ModK)
print('davies-bouldin score: ', davies_bouldin_score)
calinski_harabasz_score = calinski_harabasz(ModK)
print('calinski-harabasz score: ', calinski_harabasz_score)
dunn_score = dunn(ModK)
print('dunn score: ', dunn_score)
# %%
# We can compute this score for differents values of n_clusters.
K = range(4,8)
silhouette_scores = list()
davies_bouldin_scores = list()
calinski_harabasz_scores = list()
dunn_scores = list()
for k in K:
    ModK = ModKMeans(n_clusters=k, random_seed=42)
    ModK.fit(raw, n_jobs=5)
    
    silhouette_score = silhouette(ModK)
    silhouette_scores.append(silhouette_score)
    
    davies_bouldin_score = davies_bouldin(ModK)
    davies_bouldin_scores.append(davies_bouldin_score)
    
    calinski_harabasz_score = calinski_harabasz(ModK)
    calinski_harabasz_scores.append(calinski_harabasz_score)
    
    dunn_score = dunn(ModK)
    dunn_scores.append(dunn_score)

# %%
# We can compute this score for differents values of n_clusters.
import matplotlib.pyplot as plt

fig, axs = plt.subplots(4,1)
axs[0].plot(K,silhouette_scores)
axs[0].set_title('Silhouette')
axs[1].plot(K,davies_bouldin_scores)
axs[1].set_title('davies-bouldin')
axs[2].plot(K,calinski_harabasz_scores)
axs[2].set_title('calinski-harabasz')
axs[3].plot(K,dunn_scores)
axs[3].set_title('Dunn')

#sphinx_gallery_thumbnail_number = 2
plt.tight_layout()
plt.plot()
