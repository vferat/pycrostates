import os.path as op

import numpy as np
import mne
from mne.datasets import testing

from pycrostates.clustering import ModKMeans

data_path = testing.data_path()
fname_raw_testing = op.join(data_path, 'MEG', 'sample',
                            'sample_audvis_trunc_raw.fif')
fname_evoked_testing = op.join(data_path, 'MEG', 'sample',
                               'sample_audvis-ave.fif')

def test_ModKMeans_fit_raw():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    raw = raw.crop(0, 10)
    n_clusters = 4
    ModK = ModKMeans(n_clusters=n_clusters)
    assert ModK.n_clusters == n_clusters
    assert ModK.current_fit == 'unfitted'
    ModK.fit(raw, n_jobs=1)
    assert ModK.cluster_centers.shape == (n_clusters, len(raw.info['ch_names']))
    assert ModK.current_fit == 'Raw'
    assert ModK.GEV > 0

    raw.info['bads'] == [raw.info['ch_names'][0]]
    ModK.fit(raw, n_jobs=1)
    assert ModK.cluster_centers.shape == (n_clusters, len(raw.info['ch_names']))

def test_ModKMeans_randomstate():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    raw = raw.crop(0, 10)
    n_clusters = 4
    ModK_1 = ModKMeans(n_clusters=n_clusters, random_state=1)
    ModK_1.fit(raw, n_jobs=1)
    ModK_2 = ModKMeans(n_clusters=n_clusters, random_state=1)
    ModK_2.fit(raw, n_jobs=1)
    assert (ModK_1.cluster_centers == ModK_2.cluster_centers).all()
  
def test_ModKMeans_get_cluster_centers_as_raw():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    raw = raw.crop(0, 10)
    n_clusters = 4
    ModK = ModKMeans(n_clusters=n_clusters, random_state=1)
    ModK.fit(raw, n_jobs=1)
    raw_clusters = ModK.get_cluster_centers_as_raw()
    assert (raw_clusters.get_data().T == ModK.cluster_centers).all()


def test_ModKMeans_invert_polarity():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    raw = raw.crop(0, 10)
    n_clusters = 4
    ModK = ModKMeans(n_clusters=n_clusters, random_state=1)
    ModK.fit(raw, n_jobs=1)
    before = ModK.cluster_centers.copy()
    ModK.invert_polarity([True,True,False,False])
    after = ModK.cluster_centers
    assert (before[0] == - after[0]).all()
    assert (before[1] == - after[1]).all()
    assert (before[2] ==  after[2]).all()
    assert (before[3] ==  after[3]).all()

def test_ModKMeans_to_pickle():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    raw = raw.crop(0, 10)
    n_clusters = 4
    ModK = ModKMeans(n_clusters=n_clusters, random_state=1)
    ModK.fit(raw, n_jobs=1)
    ModK.to_pickle('file.pkl')
    import pickle
    with open('file.pkl', 'rb') as f:
        modk_load = pickle.load(f)
    assert (modk_load.cluster_centers == ModK.cluster_centers).all()
     
def test_ModKMeans_fit_epochs():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    events = mne.make_fixed_length_events(raw, 1)
    epochs = mne.epochs.Epochs(raw, events, preload=True)
    epochs = epochs.pick('eeg')
    n_clusters = 4
    ModK = ModKMeans(n_clusters=n_clusters)
    assert ModK.n_clusters == n_clusters
    assert ModK.current_fit == 'unfitted'
    ModK.fit(epochs, n_jobs=1)
    assert ModK.cluster_centers.shape == (n_clusters, len(epochs.info['ch_names']))
    assert ModK.current_fit == 'Epochs'
    assert ModK.GEV > 0

    epochs.info['bads'] == [epochs.info['ch_names'][0]]
    ModK.fit(epochs, n_jobs=1)
    assert ModK.cluster_centers.shape == (n_clusters, len(epochs.info['ch_names']))
    
def test_ModKMeans_fit_evoked():
    evoked = mne.read_evokeds(fname_evoked_testing, condition=0)
    evoked = evoked.pick('eeg')
    n_clusters = 4
    ModK = ModKMeans(n_clusters=n_clusters)
    assert ModK.n_clusters == n_clusters
    assert ModK.current_fit == 'unfitted'
    ModK.fit(evoked, n_jobs=1)
    assert ModK.cluster_centers.shape == (n_clusters, len(evoked.info['ch_names']))
    assert ModK.current_fit == 'Evoked'
    assert ModK.GEV > 0

    evoked.info['bads'] == [evoked.info['ch_names'][0]]
    ModK.fit(evoked, n_jobs=1)
    assert ModK.cluster_centers.shape == (n_clusters, len(evoked.info['ch_names']))
    
def test_BaseClustering_predict_raw():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    raw = raw.filter(0, 40)
    raw = raw.crop(0, 10)
    n_clusters = 4
    ModK = ModKMeans(n_clusters=n_clusters)
    ModK.fit(raw, n_jobs=1)
    segmentation = ModK.predict(raw)
    assert len(segmentation) == raw.get_data().shape[-1]
    assert np.isin(segmentation, np.arange(0,n_clusters+1)).all()
 
        
def test_BaseClustering_predict_evoked():
    evoked = mne.read_evokeds(fname_evoked_testing, condition=0)
    evoked = evoked.pick('eeg')
    n_clusters = 4
    ModK = ModKMeans(n_clusters=n_clusters)
    ModK.fit(evoked)
    segmentation = ModK.predict(evoked)
    assert len(segmentation) == evoked.data.shape[-1]
    assert np.isin(segmentation, np.arange(0,n_clusters+1)).all()
    
def test_BaseClustering_reorder():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    raw = raw.filter(0, 40)
    raw = raw.crop(0, 10)
    n_clusters = 4
    ModK = ModKMeans(n_clusters=n_clusters)
    ModK.fit(raw, n_jobs=1)
    cluster_centers = ModK.cluster_centers
    order = [3,2,1,0]
    ModK.reorder(order)
    assert (ModK.cluster_centers == cluster_centers[order]).all()

def test_BaseClustering_smart_reorder():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    raw = raw.filter(0, 40)
    raw = raw.crop(0, 10)
    n_clusters = 4
    ModK = ModKMeans(n_clusters=n_clusters)
    ModK.fit(raw, n_jobs=1)
    cluster_centers = ModK.cluster_centers
    ModK.smart_reorder()
    assert ModK.cluster_centers.shape == cluster_centers.shape
    assert np.isin(ModK.cluster_centers, cluster_centers).all()
