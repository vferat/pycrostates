"""Test import/export to FIFF format."""

import os
from pathlib import Path

from mne.datasets import testing
from mne.io import read_raw_fif
import numpy as np
import pytest

from pycrostates.cluster import ModKMeans
from pycrostates.io.fiff import write_cluster, read_cluster
from pycrostates.utils._logs import logger, set_log_level


set_log_level('INFO')
logger.propagate = True


# Fit one for general purposes
directory = Path(testing.data_path()) / 'MEG' / 'sample'
fname = directory / 'sample_audvis_trunc_raw.fif'
raw = read_raw_fif(fname, preload=True)
raw2 = raw.copy().crop(10, None).apply_proj()
raw = raw.crop(0, 10).apply_proj()
ModK = ModKMeans(n_clusters=4, n_init=10, max_iter=100, tol=1e-4,
                 random_state=1)
ModK.fit(raw, picks='eeg', n_jobs=1)


def test_write_and_read(tmp_path, caplog):
    """Test writing of .fif file."""
    # writing to .fif
    fname1 = tmp_path / 'cluster.fif'
    caplog.clear()
    write_cluster(
        fname1,
        ModK._cluster_centers_,
        ModK._info,
        'ModKMeans',
        ModK._cluster_names,
        ModK._fitted_data,
        ModK._labels_,
        n_init=ModK._n_init,
        max_iter=ModK._max_iter,
        tol=ModK._tol,
        GEV_=ModK._GEV_
        )
    assert 'Writing clustering solution' in caplog.text

    # writing to .gz (compression)
    fname2 = tmp_path / 'cluster.fif.gz'
    caplog.clear()
    write_cluster(
        fname2,
        ModK._cluster_centers_,
        ModK._info,
        'ModKMeans',
        ModK._cluster_names,
        ModK._fitted_data,
        ModK._labels_,
        n_init=ModK._n_init,
        max_iter=ModK._max_iter,
        tol=ModK._tol,
        GEV_=ModK._GEV_
        )
    assert 'Writing clustering solution' in caplog.text

    # compare size
    assert os.path.getsize(fname2) < os.path.getsize(fname1)

    # re-load the 2 saved files
    caplog.clear()
    ModK1 = read_cluster(fname1)
    assert 'Reading clustering solution' in caplog.text
    caplog.clear()
    ModK2 = read_cluster(fname2)
    assert 'Reading clustering solution' in caplog.text

    # compare
    assert ModK == ModK1
    assert ModK == ModK2
    assert ModK1 == ModK2  # sanity-check

    # test prediction
    segmentation = ModK.predict(raw2, picks='eeg')
    segmentation1 = ModK1.predict(raw2, picks='eeg')
    segmentation2 = ModK2.predict(raw2, picks='eeg')

    assert np.allclose(segmentation.labels, segmentation1.labels)
    assert np.allclose(segmentation.labels, segmentation2.labels)
    assert np.allclose(segmentation1.labels, segmentation2.labels)
