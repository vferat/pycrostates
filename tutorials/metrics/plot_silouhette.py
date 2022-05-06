"""
Choosing the number of cluster centers k.
=========================================

In this tutorial, we will see how to get indications on the best
number of clusters centers k to use.
It is important to note that there is no single solution to this choice.
It will always be a trade off between the quality of the clustering,
the possibility to analyze the microstates in the light of the literature
and the variance expressed by the decomposition
"""

#%%
# We first start by importing some EEG data

from mne.io import read_raw_eeglab

from pycrostates.cluster import ModKMeans
from pycrostates.datasets import lemon

raw_fname = lemon.load_data(subject_id="010017", condition="EC")
raw = read_raw_eeglab(raw_fname, preload=True)
raw.crop(0, 30)

raw.pick("eeg")
raw.set_eeg_reference("average")

#%%
# In this exemple, we will used the [Silhouette Coefficient](https://doi.org/10.1016/0377-0427(87)90125-7)
# to evaluate the
# Pycrostates implementations relies on sklearn implementation
# while ensuring that the metric used for distance computations is
# the opposite value of the absolute spatial correlation.
# More details can be found in [sklearn User Guide] (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score)
from pycrostates.metrics.metrics import silhouette

cluster_numbers = range(2,12)
silhouette_scores = []
for k in range(2,12):
    ModK = ModKMeans(n_clusters=k, random_state=42)
    ModK.fit(raw, n_jobs=12)
    score = silhouette(ModK)
    silhouette_scores.append(score)
#%%
# Most methods need the modified Kmeans to be fitted. This can be done with
# either :class:`mne.io.Raw` or :class:`mne.epochs.Epochs` data structures.
# Note that, depending on your setup, you can change ``n_jobs=1`` in order to
# use parallel processing and reduce computation time.
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(cluster_numbers, silhouette_scores)
plt.xlabel('n_clusters')
plt.ylabel('Silhouette score') 
plt.show()