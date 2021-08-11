import os.path as op

import numpy as np
import mne
from mne.datasets import testing

from pycrostates.clustering import ModKMeans
from pycrostates.segmentation import RawSegmentation, EpochsSegmentation

data_path = testing.data_path()
fname_raw_testing = op.join(data_path, 'MEG', 'sample',
                            'sample_audvis_trunc_raw.fif')

def test_ModKMeans_randomstate():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    raw = raw.crop(0, 10)
    n_clusters = 4
    ModK_1 = ModKMeans(n_clusters=n_clusters, random_state=1)
    ModK_1.fit(raw, n_jobs=1)
    ModK_2 = ModKMeans(n_clusters=n_clusters, random_state=1)
    ModK_2.fit(raw, n_jobs=1)
    assert (ModK_1.cluster_centers_ == ModK_2.cluster_centers_).all()

def test_ModKMeans_get_cluster_centers_as_raw():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    raw = raw.crop(0, 10)
    n_clusters = 4
    ModK = ModKMeans(n_clusters=n_clusters, random_state=1)
    ModK.fit(raw, n_jobs=1)
    raw_clusters = ModK.get_cluster_centers_as_raw()
    assert (raw_clusters.get_data().T == ModK.cluster_centers_).all()

def test_ModKMeans_invert_polarity():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    raw = raw.crop(0, 10)
    n_clusters = 4
    ModK = ModKMeans(n_clusters=n_clusters, random_state=1)
    ModK.fit(raw, n_jobs=1)
    before = ModK.cluster_centers_.copy()
    ModK.invert_polarity([True,True,False,False])
    after = ModK.cluster_centers_
    assert (before[0] == - after[0]).all()
    assert (before[1] == - after[1]).all()
    assert (before[2] ==  after[2]).all()
    assert (before[3] ==  after[3]).all()

def test_BaseClustering_reorder():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    raw = raw.filter(0, 40)
    raw = raw.crop(0, 10)
    n_clusters = 4
    ModK = ModKMeans(n_clusters=n_clusters)
    ModK.fit(raw, n_jobs=1)
    cluster_centers = ModK.cluster_centers_
    order = [3,2,1,0]
    ModK.reorder(order)
    assert (ModK.cluster_centers_ == cluster_centers[order]).all()

def test_BaseClustering_smart_reorder():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    raw = raw.filter(0, 40)
    raw = raw.crop(0, 10)
    n_clusters = 4
    ModK = ModKMeans(n_clusters=n_clusters)
    ModK.fit(raw, n_jobs=1)
    cluster_centers = ModK.cluster_centers_
    ModK.smart_reorder()
    assert ModK.cluster_centers_.shape == cluster_centers.shape
    assert np.isin(ModK.cluster_centers_, cluster_centers).all()

def test_ModKMeans_fit_raw():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    raw = raw.crop(0, 10)
    n_clusters = 4
    ModK = ModKMeans(n_clusters=n_clusters)
    assert ModK.n_clusters == n_clusters
    assert ModK.current_fit == 'unfitted'
    ModK.fit(raw, n_jobs=1)
    assert ModK.cluster_centers_.shape == (n_clusters, len(raw.info['ch_names']))
    assert ModK.current_fit == 'Raw'
    assert ModK.GEV_ > 0

    raw.info['bads'] = [raw.info['ch_names'][0]]
    ModK.fit(raw, n_jobs=1)
    assert ModK.cluster_centers_.shape == (n_clusters, len(raw.info['ch_names']))

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
    assert ModK.cluster_centers_.shape == (n_clusters, len(epochs.info['ch_names']))
    assert ModK.current_fit == 'Epochs'
    assert ModK.GEV_ > 0

    epochs.info['bads'] = [epochs.info['ch_names'][0]]
    ModK.fit(epochs, n_jobs=1, min_peak_distance=3)
    assert ModK.cluster_centers_.shape == (n_clusters, len(epochs.info['ch_names']))

    epochs.info['bads'] = [epochs.info['ch_names'][0]]
    ModK.fit(epochs, n_jobs=1)
    assert ModK.cluster_centers_.shape == (n_clusters, len(epochs.info['ch_names']))
 
def test_ModKMeans_fit_min_peak_distance():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    raw = raw.crop(0, 10)
    n_clusters = 4
    ModK = ModKMeans(n_clusters=n_clusters)
    ModK.fit(raw,
             min_peak_distance=3)
    assert ModK.fitted_data_.shape[-1] <= raw.get_data().shape[-1] / 3
    
def test_ModKMeans_fit_start_stop():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    raw = raw.crop(0, 10)
    n_clusters = 4
    ModK = ModKMeans(n_clusters=n_clusters)
    ModK.fit(raw,
             start=1,
             stop=9)
    assert ModK.fitted_data_.shape[-1] == raw.get_data(start=1,stop=9).shape[-1]
    
def test_ModKMeans_fit_reject_by_annotation():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    raw = raw.crop(0, 10)
    bad_annot = mne.Annotations([1], [2], 'bad')
    raw.set_annotations(bad_annot)
    n_clusters = 4
    ModK = ModKMeans(n_clusters=n_clusters)
    ModK.fit(raw)
    assert ModK.fitted_data_.shape[-1] == raw.get_data(reject_by_annotation='omit').shape[-1]

def test_BaseClustering_predict_raw():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    raw = raw.filter(0, 40)
    raw = raw.crop(0, 10)
    n_clusters = 4
    ModK = ModKMeans(n_clusters=n_clusters)
    ModK.fit(raw, n_jobs=1)
    segmentation = ModK.predict(raw)
    assert isinstance(segmentation, RawSegmentation)
 

def test_BaseClustering_predict_epochs():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    raw = raw.filter(0, 40)
    raw = raw.crop(0, 10)
    events = mne.make_fixed_length_events(raw, 1)
    epochs = mne.epochs.Epochs(raw, events, preload=True)
    epochs = epochs.pick('eeg')
    n_clusters = 4
    ModK = ModKMeans(n_clusters=n_clusters)
    ModK.fit(epochs, n_jobs=1)
    segmentation = ModK.predict(epochs)
    assert isinstance(segmentation, EpochsSegmentation)
