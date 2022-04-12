import os
from pathlib import Path

from mne.datasets import testing
from mne.io import read_raw_fif
import pytest

from pycrostates import __version__
from pycrostates.cluster import ModKMeans
from pycrostates.io import read_cluster
from pycrostates.utils._logs import logger, set_log_level


set_log_level('INFO')
logger.propagate = True


directory = Path(testing.data_path()) / 'MEG' / 'sample'
fname = directory / 'sample_audvis_trunc_raw.fif'
raw = read_raw_fif(fname, preload=True)
raw = raw.crop(0, 10).apply_proj()
ModK = ModKMeans(n_clusters=4, n_init=10, max_iter=100, tol=1e-4,
                 random_state=1)
ModK.fit(raw, picks='eeg')


def test_reader(tmp_path, caplog):
    """Test general reader."""
    ModK.save(tmp_path / 'cluster.fif')
    ModK.save(tmp_path / 'cluster2.fif')
    os.rename(tmp_path / 'cluster2.fif', tmp_path / 'cluster')
    ModK.save(tmp_path / 'cluster.fif.gz')

    # load
    caplog.clear()
    ModK1 = read_cluster(tmp_path / 'cluster.fif')
    assert f"saved with pycrostates '{__version__}'." in caplog.text
    assert ModK == ModK1
    caplog.clear()
    ModK2 = read_cluster(tmp_path / 'cluster.fif.gz')
    assert f"saved with pycrostates '{__version__}'." in caplog.text
    assert ModK == ModK2

    # invalid file name
    with pytest.raises(ValueError, match='File format is not supported.'):
        read_cluster(tmp_path / 'cluster')
