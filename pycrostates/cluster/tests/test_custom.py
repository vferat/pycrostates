import mne
from pycrostates.cluster import Custom, ModKMeans
from pycrostates.preprocessing import extract_gfp_peaks
import matplotlib.pyplot as plt
import seaborn as sns
from mne.io import read_raw_eeglab
import numpy as np

from pycrostates.cluster import ModKMeans
from pycrostates.datasets import lemon
from pycrostates.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    dunn_score,
    davies_bouldin_score,
)

if __name__ == "__main__":
    # U=[1,1,1,0,1]
    # V=[2,2,2,1,2]
    # print(custom._spatial_corr(U,V))
    
    # generate some data to test
    # random_array = np.random.rand(100,19)
    # random_center = np.random.rand(4,19)
    # values = [0,1,2,3]
    # random_label = np.random.choice(values, size=100)
    # print(random_label.shape)



    # load sample dataset
    raw_fname = lemon.data_path(subject_id='010017', condition='EC')
    raw = read_raw_eeglab(raw_fname, preload=True)
    raw.crop(0, 10)  # crop the dataset to speed up computation
    raw.pick('eeg')  # select EEG channels
    raw.set_eeg_reference('average')  # Apply a common average reference


    # extract gfp
    # gfp_data = extract_gfp_peaks(raw)
    # ModK = ModKMeans(n_clusters=4, random_state=42)
    # ModK.fit(gfp_data, n_jobs=6, verbose="WARNING")
    # print(ModK.GEV_)
    # print(ModK.cluster_centers_.shape)
    # print(ModK.labels_.shape)
    # custom.assign_cluster_centers(ModK.cluster_centers_)
    # custom.assign_labels(ModK.labels_)
    # custom = Custom(4)
    # custom.fit(gfp_data, ModK.cluster_centers_, ModK.labels_)
    # print(custom.GEV_)





    cluster_numbers = range(2, 9)
    scores = {
        "Silhouette": np.zeros(len(cluster_numbers)),
        "Calinski-Harabasaz": np.zeros(len(cluster_numbers)),
        "Dunn": np.zeros(len(cluster_numbers)),
        "Davies-Bouldin": np.zeros(len(cluster_numbers)),
    }
    gfp_data = extract_gfp_peaks(raw)
    for k, n_clusters in enumerate(cluster_numbers):
        # fit K-means algorithm with a set number of cluster centers
        ModK = ModKMeans(n_clusters=n_clusters, random_state=42)
        ModK.fit(gfp_data, n_jobs=2, verbose="WARNING")

        custom = Custom(n_clusters=n_clusters)
        custom.fit(gfp_data, ModK.cluster_centers_, ModK.labels_)
        
        # compute scores
        scores["Silhouette"][k] = silhouette_score(custom)
        scores["Calinski-Harabasaz"][k] = calinski_harabasz_score(custom)
        scores["Dunn"][k] = dunn_score(custom)
        scores["Davies-Bouldin"][k] = davies_bouldin_score(custom)

    f, ax = plt.subplots(2, 2, sharex=True)
    for k, (score, values) in enumerate(scores.items()):
        ax[k // 2, k % 2].bar(x=cluster_numbers, height=values)
        ax[k // 2, k % 2].set_title(score)
    plt.text(
        0.03, 0.5, "Score",
        horizontalalignment='center',
        verticalalignment='center',
        rotation=90,
        fontdict=dict(size=14),
        transform=f.transFigure,
    )
    plt.text(
        0.5, 0.03, "Number of clusters",
        horizontalalignment='center',
        verticalalignment='center',
        fontdict=dict(size=14),
        transform=f.transFigure,
    )
    plt.show()
    # print(random_array.shape)
    # print(random_label)
    # print(random_center[random_label])
    # custom._cal_GEV(gfp_data.get_data())
#     segmentation = custom.predict(
#     raw,
#     reject_by_annotation=True,
#     factor=10,
#     half_window_size=10,
#     min_segment_length=5,
#     reject_edges=True,
# )
#     print(segmentation.labels)
#     print(segmentation.labels.shape)
#     segmentation.plot(tmin=1, tmax=5)
#     plt.show()
#     parameters = segmentation.compute_parameters()

#     x = ModK.cluster_names

#     y = [parameters[elt + "_gev"] for elt in x]

#     ax = sns.barplot(x=x, y=y)
#     ax.set_xlabel("Microstates")
#     ax.set_ylabel("Global explained Variance (ratio)")
#     plt.show()
    # print(gfp_data.get_data().shape)
    # print(custom.GEV_)
    # custom.assign_labels(1)
    
