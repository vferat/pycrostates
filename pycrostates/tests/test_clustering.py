import os.path as op

import pytest

import numpy as np
import mne
from mne.datasets import testing

from pycrostates.clustering import ModKMeans

data_path = testing.data_path(download=True)
fname_raw_testing = op.join(data_path, 'MEG', 'sample',
                            'sample_audvis_trunc_raw.fif')

def test_ModKMeans_fit():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    raw = raw.filter(0, 40)
    raw = raw.crop(0, 10)
    n_clusters = 4
    ModK = ModKMeans(n_clusters=n_clusters)
    assert ModK.n_clusters == n_clusters
    assert ModK.current_fit is False
    assert ModK.GEV is None
    assert ModK.Info is None
    ModK.fit(raw, n_jobs=1)
    assert len(ModK.cluster_centers) == n_clusters
    assert ModK.current_fit is True
    assert ModK.GEV > 0
    assert ModK.info == raw.info
    
def test_BaseClustering_transform():
    raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
    raw = raw.pick('eeg')
    raw = raw.filter(0, 40)
    raw = raw.crop(0, 10)
    n_clusters = 4
    ModK = ModKMeans(n_clusters=n_clusters)
    ModK.fit(raw, n_jobs=1)
    distances = ModK.transform(raw)
    assert len(distances) == raw.get_data().shape[-1]
    assert (distances > 0).all()
    
def test_BaseClustering_predict():
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
