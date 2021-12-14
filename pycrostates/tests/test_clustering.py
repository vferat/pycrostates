import os.path as op
import itertools

import numpy as np
import mne
from mne.datasets import testing

from pycrostates.clustering import ModKMeans
from pycrostates.segmentation import RawSegmentation, EpochsSegmentation

n_clusters = 4

data_path = testing.data_path()
fname_raw_testing = op.join(data_path, 'MEG', 'sample',
                            'sample_audvis_trunc_raw.fif')
raw = mne.io.read_raw_fif(fname_raw_testing, preload=True)
raw = raw.pick('eeg')
raw = raw.crop(0, 10)
events = mne.make_fixed_length_events(raw, 1)
epochs = mne.epochs.Epochs(raw, events, preload=True)


def test_ModKMeans_randomstate():
    ModK_1 = ModKMeans(n_clusters=n_clusters, random_state=1)
    ModK_1.fit(raw, n_jobs=1)
    ModK_2 = ModKMeans(n_clusters=n_clusters, random_state=1)
    ModK_2.fit(raw, n_jobs=1)
    assert (ModK_1.cluster_centers_ == ModK_2.cluster_centers_).all()


def test_ModKMeans_copy():
    ModK_1 = ModKMeans(n_clusters=n_clusters, random_state=1)
    ModK_2 = ModK_1.copy()
    ModK_1.n_clusters = 12
    assert ModK_2.n_clusters == 4


def test_ModKMeans_repr():
    ModK = ModKMeans(n_clusters=n_clusters, random_state=1)
    assert isinstance(ModK.__repr__(), str)


def test_ModKMeans_get_cluster_centers():
    ModK = ModKMeans(n_clusters=n_clusters, random_state=1)
    ModK.fit(raw, n_jobs=1)
    raw_clusters = ModK.get_cluster_centers()
    assert (raw_clusters == ModK.cluster_centers_).all()


def test_ModKMeans_get_cluster_centers_as_raw():
    ModK = ModKMeans(n_clusters=n_clusters, random_state=1)
    ModK.fit(raw, n_jobs=1)
    raw_clusters = ModK.get_cluster_centers_as_raw()
    assert (raw_clusters.get_data().T == ModK.cluster_centers_).all()


def test_ModKMeans_rename_clusters():
    names = ['a', 'b', 'c', 'd']
    ModK = ModKMeans(n_clusters=n_clusters, random_state=1)
    ModK.fit(raw, n_jobs=1)
    ModK.rename_clusters(names)
    assert np.all(ModK.names == names)


def test_ModKMeans_invert_polarity():
    ModK = ModKMeans(n_clusters=n_clusters, random_state=1)
    ModK.fit(raw, n_jobs=1)
    before = ModK.cluster_centers_.copy()
    ModK.invert_polarity([True, True, False, False])
    after = ModK.cluster_centers_
    assert (before[0] == - after[0]).all()
    assert (before[1] == - after[1]).all()
    assert (before[2] == after[2]).all()
    assert (before[3] == after[3]).all()


def test_BaseClustering_reorder():
    ModK = ModKMeans(n_clusters=n_clusters)
    ModK.fit(raw, n_jobs=1)
    cluster_centers = ModK.cluster_centers_
    order = [3, 2, 1, 0]
    ModK.reorder(order)
    assert (ModK.cluster_centers_ == cluster_centers[order]).all()


def test_ModKMeans_fit_raw():
    ModK = ModKMeans(n_clusters=n_clusters)
    assert ModK.n_clusters == n_clusters
    assert ModK.current_fit == 'unfitted'
    ModK.fit(raw, n_jobs=1)
    assert ModK.cluster_centers_.shape == \
        (n_clusters, len(raw.info['ch_names']))
    assert ModK.current_fit == 'Raw'
    assert ModK.GEV_ > 0

    raw.info['bads'] = [raw.info['ch_names'][0]]
    ModK.fit(raw, n_jobs=1)
    assert ModK.cluster_centers_.shape == \
        (n_clusters, len(raw.info['ch_names']))


def test_ModKMeans_fit_epochs():
    ModK = ModKMeans(n_clusters=n_clusters)
    assert ModK.n_clusters == n_clusters
    assert ModK.current_fit == 'unfitted'
    ModK.fit(epochs, n_jobs=1)
    assert ModK.cluster_centers_.shape == \
        (n_clusters, len(epochs.info['ch_names']))
    assert ModK.current_fit == 'Epochs'
    assert ModK.GEV_ > 0

    epochs.info['bads'] = [epochs.info['ch_names'][0]]
    ModK.fit(epochs, n_jobs=1)
    assert ModK.cluster_centers_.shape == \
        (n_clusters, len(epochs.info['ch_names']))


def test_ModKMeans_fit_n_jobs():
    ModK = ModKMeans(n_clusters=n_clusters)
    ModK.fit(raw, n_jobs=2)
    assert ModK.current_fit == 'Raw'


def test_ModKMeans_fit_start_stop():
    ModK = ModKMeans(n_clusters=n_clusters)
    ModK.fit(raw,
             start=1,
             stop=9)
    assert ModK.fitted_data_.shape[-1] == \
        raw.get_data(start=1, stop=9).shape[-1]


def test_ModKMeans_fit_reject_by_annotation():
    bad_annot = mne.Annotations([1], [2], 'bad')
    raw_ = raw.copy()
    raw_.set_annotations(bad_annot)
    ModK = ModKMeans(n_clusters=n_clusters)
    ModK.fit(raw_)
    assert ModK.fitted_data_.shape[-1] == raw_.get_data(
        reject_by_annotation='omit').shape[-1]


def test_BaseClustering_predict_raw():
    ModK = ModKMeans(n_clusters=n_clusters)
    ModK.fit(raw, n_jobs=1)
    segmentation = ModK.predict(raw,
                                factor=0,
                                rejected_first_last_segments=False,
                                min_segment_lenght=0,
                                reject_by_annotation=False)
    assert isinstance(segmentation, RawSegmentation)


def test_BaseClustering_predict_epochs():
    ModK = ModKMeans(n_clusters=n_clusters)
    ModK.fit(epochs)
    segmentation = ModK.predict(epochs,
                                factor=0,
                                rejected_first_last_segments=False,
                                min_segment_lenght=0,
                                reject_by_annotation=False)
    assert isinstance(segmentation, EpochsSegmentation)


def test_BaseClustering_predict_raw_rejected_first_last_segments():
    ModK = ModKMeans(n_clusters=n_clusters)
    ModK.fit(raw, n_jobs=1)
    segmentation = ModK.predict(raw, rejected_first_last_segments=True)
    assert segmentation.segmentation[0] == 0
    assert segmentation.segmentation[-1] == 0


def test_BaseClustering_predict_epochs_rejected_first_last_segments():
    ModK = ModKMeans(n_clusters=n_clusters)
    ModK.fit(epochs, n_jobs=1)
    segmentation = ModK.predict(epochs, rejected_first_last_segments=True)
    for epoch_seg in segmentation.segmentation:
        assert epoch_seg[0] == 0
        assert epoch_seg[-1] == 0


def test_BaseClustering_predict_raw_min_segment_lenght():
    ModK = ModKMeans(n_clusters=n_clusters)
    ModK.fit(raw, n_jobs=1)
    segmentation = ModK.predict(raw, min_segment_lenght=3)
    segment_lengths = [
        len(list(group))
        for _, group in itertools.groupby(segmentation.segmentation)][1:-2]
    assert np.all(np.array(segment_lengths) > 3)


def test_BaseClustering_predict_raw_smoothing():
    ModK = ModKMeans(n_clusters=n_clusters)
    ModK.fit(raw, n_jobs=1)
    segmentation = ModK.predict(raw, factor=10)
    assert isinstance(segmentation, RawSegmentation)
