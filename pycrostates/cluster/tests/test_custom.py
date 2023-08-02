import mne
from pycrostates.cluster import Custom, ModKMeans
from pycrostates.preprocessing import extract_gfp_peaks
import matplotlib.pyplot as plt
import seaborn as sns
from mne.io import read_raw_eeglab

from pycrostates.cluster import ModKMeans
from pycrostates.datasets import lemon

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
    gfp_data = extract_gfp_peaks(raw)
    ModK = ModKMeans(n_clusters=4, random_state=42)
    ModK.fit(gfp_data, n_jobs=6, verbose="WARNING")
    # print(ModK.GEV_)
    # print(ModK.cluster_centers_.shape)
    # print(ModK.labels_.shape)
    # custom.assign_cluster_centers(ModK.cluster_centers_)
    # custom.assign_labels(ModK.labels_)
    custom = Custom(4)
    custom.fit(gfp_data, ModK.cluster_centers_, ModK.labels_)
    print(custom.GEV_)

    # print(random_array.shape)
    # print(random_label)
    # print(random_center[random_label])
    # custom._cal_GEV(gfp_data.get_data())
    segmentation = custom.predict(
    raw,
    reject_by_annotation=True,
    factor=10,
    half_window_size=10,
    min_segment_length=5,
    reject_edges=True,
)
    print(segmentation.labels)
    print(segmentation.labels.shape)
    segmentation.plot(tmin=1, tmax=5)
    plt.show()
    parameters = segmentation.compute_parameters()

    x = ModK.cluster_names

    y = [parameters[elt + "_gev"] for elt in x]

    ax = sns.barplot(x=x, y=y)
    ax.set_xlabel("Microstates")
    ax.set_ylabel("Global explained Variance (ratio)")
    plt.show()
    # print(gfp_data.get_data().shape)
    # print(custom.GEV_)
    # custom.assign_labels(1)
    
