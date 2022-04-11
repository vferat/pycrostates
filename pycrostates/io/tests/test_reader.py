import os
from pathlib import Path

from mne.datasets import testing
from mne.io import read_raw_fif
import pytest

from pycrostates.cluster import ModKMeans
from pycrostates.io import read_cluster


directory = Path(testing.data_path()) / 'MEG' / 'sample'
fname = directory / 'sample_audvis_trunc_raw.fif'
raw = read_raw_fif(fname, preload=True)
raw = raw.crop(0, 10).apply_proj()
ModK = ModKMeans(n_clusters=4, n_init=10, max_iter=100, tol=1e-4,
                 random_state=1)
ModK.fit(raw, picks='eeg')


def test_reader(tmp_path):
    """Test general reader."""
    ModK.save(tmp_path / 'cluster.fif')
    ModK.save(tmp_path / 'cluster2.fif')
    os.rename(tmp_path / 'cluster2.fif', tmp_path / 'cluster')
    ModK.save(tmp_path / 'cluster.fif.gz')

    # load
    ModK1 = read_cluster(tmp_path / 'cluster.fif')
    assert ModK == ModK1
    ModK2 = read_cluster(tmp_path / 'cluster.fif.gz')
    assert ModK == ModK2

    # invalid file name
    with pytest.raises(ValueError, match='File format is not supported.'):
        read_cluster(tmp_path / 'cluster')
