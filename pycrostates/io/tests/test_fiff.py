"""Test import/export to FIFF format."""

import os

import numpy as np
import pytest
from mne.datasets import testing
from mne.io import read_raw_fif
from mne.preprocessing import ICA

from pycrostates import __version__
from pycrostates.cluster import ModKMeans
from pycrostates.io.fiff import _prepare_kwargs, _read_cluster, _write_cluster
from pycrostates.utils._logs import logger, set_log_level

set_log_level("INFO")
logger.propagate = True


# Fit one for general purposes
directory = testing.data_path() / "MEG" / "sample"
fname = directory / "sample_audvis_trunc_raw.fif"
raw = read_raw_fif(fname, preload=True)
raw2 = raw.copy().crop(10, None).apply_proj()
raw = raw.crop(0, 10).apply_proj()
ModK = ModKMeans(
    n_clusters=4, n_init=10, max_iter=100, tol=1e-4, random_state=1
)
ModK.fit(raw, picks="eeg", n_jobs=1)


def test_write_and_read(tmp_path, caplog):
    """Test writing of .fif file."""
    # writing to .fif
    fname1 = tmp_path / "cluster.fif"
    caplog.clear()
    _write_cluster(
        fname1,
        ModK._cluster_centers_,
        ModK._info,
        "ModKMeans",
        ModK._cluster_names,
        ModK._fitted_data,
        ModK._labels_,
        n_init=ModK._n_init,
        max_iter=ModK._max_iter,
        tol=ModK._tol,
        GEV_=ModK._GEV_,
    )
    assert "Writing clustering solution" in caplog.text

    # writing to .gz (compression)
    fname2 = tmp_path / "cluster.fif.gz"
    caplog.clear()
    _write_cluster(
        fname2,
        ModK._cluster_centers_,
        ModK._info,
        "ModKMeans",
        ModK._cluster_names,
        ModK._fitted_data,
        ModK._labels_,
        n_init=ModK.n_init,
        max_iter=ModK.max_iter,
        tol=ModK.tol,
        GEV_=ModK.GEV_,
    )
    assert "Writing clustering solution" in caplog.text

    # compare size
    assert os.path.getsize(fname2) < os.path.getsize(fname1)

    # re-load the 2 saved files
    caplog.clear()
    ModK1, version = _read_cluster(fname1)
    assert "Reading clustering solution" in caplog.text
    assert version == __version__
    caplog.clear()
    ModK2, version = _read_cluster(fname2)
    assert "Reading clustering solution" in caplog.text
    assert version == __version__

    # compare
    assert ModK == ModK1
    assert ModK == ModK2
    assert ModK1 == ModK2  # sanity-check

    # test prediction
    segmentation = ModK.predict(raw2)
    segmentation1 = ModK1.predict(raw2)
    segmentation2 = ModK2.predict(raw2)

    assert np.allclose(segmentation._labels, segmentation1._labels)
    assert np.allclose(segmentation._labels, segmentation2._labels)
    assert np.allclose(segmentation1._labels, segmentation2._labels)


def test_invalid_write(tmp_path):
    """Test invalid arguments provided to write."""
    with pytest.raises(TypeError, match="'fname' must be an instance of"):
        _write_cluster(
            101,
            ModK._cluster_centers_,
            ModK._info,
            "ModKMeans",
            ModK._cluster_names,
            ModK._fitted_data,
            ModK._labels_,
            n_init=ModK._n_init,
            max_iter=ModK._max_iter,
            tol=ModK._tol,
            GEV_=ModK._GEV_,
        )

    with pytest.raises(
        TypeError, match="'cluster_centers_' must be an instance of"
    ):
        _write_cluster(
            tmp_path / "cluster.fif",
            list(ModK._cluster_centers_),
            ModK._info,
            "ModKMeans",
            ModK._cluster_names,
            ModK._fitted_data,
            ModK._labels_,
            n_init=ModK._n_init,
            max_iter=ModK._max_iter,
            tol=ModK._tol,
            GEV_=ModK._GEV_,
        )

    with pytest.raises(
        ValueError, match="Argument 'cluster_centers_' should be a 2D"
    ):
        _write_cluster(
            tmp_path / "cluster.fif",
            ModK._cluster_centers_.flatten(),
            ModK._info,
            "ModKMeans",
            ModK._cluster_names,
            ModK._fitted_data,
            ModK._labels_,
            n_init=ModK._n_init,
            max_iter=ModK._max_iter,
            tol=ModK._tol,
            GEV_=ModK._GEV_,
        )

    with pytest.raises(TypeError, match="'chinfo' must be an instance of"):
        _write_cluster(
            tmp_path / "cluster.fif",
            ModK._cluster_centers_,
            101,
            "ModKMeans",
            ModK._cluster_names,
            ModK._fitted_data,
            ModK._labels_,
            n_init=ModK.n_init,
            max_iter=ModK.max_iter,
            tol=ModK.tol,
            GEV_=ModK.GEV_,
        )

    with pytest.raises(TypeError, match="'algorithm' must be an instance of"):
        _write_cluster(
            tmp_path / "cluster.fif",
            ModK._cluster_centers_,
            ModK._info,
            101,
            ModK._cluster_names,
            ModK._fitted_data,
            ModK._labels_,
            n_init=ModK._n_init,
            max_iter=ModK._max_iter,
            tol=ModK._tol,
            GEV_=ModK._GEV_,
        )

    with pytest.raises(
        ValueError, match="Invalid value for the 'algorithm' parameter"
    ):
        _write_cluster(
            tmp_path / "cluster.fif",
            ModK._cluster_centers_,
            ModK._info,
            "101",
            ModK._cluster_names,
            ModK._fitted_data,
            ModK._labels_,
            n_init=ModK._n_init,
            max_iter=ModK._max_iter,
            tol=ModK._tol,
            GEV_=ModK._GEV_,
        )

    with pytest.raises(
        TypeError, match="'cluster_names' must be an instance of"
    ):
        _write_cluster(
            tmp_path / "cluster.fif",
            ModK._cluster_centers_,
            ModK._info,
            "ModKMeans",
            tuple(ModK._cluster_names),
            ModK._fitted_data,
            ModK._labels_,
            n_init=ModK._n_init,
            max_iter=ModK._max_iter,
            tol=ModK._tol,
            GEV_=ModK._GEV_,
        )

    with pytest.raises(
        ValueError, match="Argument 'cluster_names' and 'cluster_centers_'"
    ):
        _write_cluster(
            tmp_path / "cluster.fif",
            ModK._cluster_centers_,
            ModK._info,
            "ModKMeans",
            ModK._cluster_names[0:-2],
            ModK._fitted_data,
            ModK._labels_,
            n_init=ModK._n_init,
            max_iter=ModK._max_iter,
            tol=ModK._tol,
            GEV_=ModK._GEV_,
        )

    with pytest.raises(
        TypeError, match="'fitted_data' must be an instance of"
    ):
        _write_cluster(
            tmp_path / "cluster.fif",
            ModK._cluster_centers_,
            ModK._info,
            "ModKMeans",
            ModK._cluster_names,
            list(ModK._fitted_data),
            ModK._labels_,
            n_init=ModK._n_init,
            max_iter=ModK._max_iter,
            tol=ModK._tol,
            GEV_=ModK._GEV_,
        )

    with pytest.raises(
        ValueError, match="Argument 'fitted_data' should be a 2D"
    ):
        _write_cluster(
            tmp_path / "cluster.fif",
            ModK._cluster_centers_,
            ModK._info,
            "ModKMeans",
            ModK._cluster_names,
            ModK._fitted_data.flatten(),
            ModK._labels_,
            n_init=ModK._n_init,
            max_iter=ModK._max_iter,
            tol=ModK._tol,
            GEV_=ModK._GEV_,
        )

    with pytest.raises(TypeError, match="'labels_' must be an instance of"):
        _write_cluster(
            tmp_path / "cluster.fif",
            ModK._cluster_centers_,
            ModK._info,
            "ModKMeans",
            ModK._cluster_names,
            ModK._fitted_data,
            list(ModK._labels_),
            n_init=ModK._n_init,
            max_iter=ModK._max_iter,
            tol=ModK._tol,
            GEV_=ModK._GEV_,
        )

    with pytest.raises(
        ValueError, match="Argument 'labels_' should be a 1D array."
    ):
        _write_cluster(
            tmp_path / "cluster.fif",
            ModK._cluster_centers_,
            ModK._info,
            "ModKMeans",
            ModK._cluster_names,
            ModK._fitted_data,
            ModK._labels_.reshape(2, ModK._labels_.size // 2),
            n_init=ModK._n_init,
            max_iter=ModK._max_iter,
            tol=ModK._tol,
            GEV_=ModK._GEV_,
        )


def test_prepare_kwargs():
    """Test _prepare_kwargs."""
    # working
    kwargs = dict(
        n_init=ModK._n_init,
        max_iter=ModK._max_iter,
        tol=ModK._tol,
        GEV_=ModK._GEV_,
    )
    _prepare_kwargs("ModKMeans", kwargs)

    # remove one value
    kwargs = dict(max_iter=ModK._max_iter, tol=ModK._tol, GEV_=ModK._GEV_)
    with pytest.raises(ValueError, match="Wrong kwargs provided for"):
        _prepare_kwargs("ModKMeans", kwargs)


def test_prepare_kwargs_ModKMeans():
    """Test invalid key/values for ModKMeans."""
    kwargs = dict(
        n_init=-101, max_iter=ModK._max_iter, tol=ModK._tol, GEV_=ModK._GEV_
    )
    with pytest.raises(
        ValueError, match="initialization must be a positive integer"
    ):
        _prepare_kwargs("ModKMeans", kwargs)

    kwargs = dict(
        n_init=ModK._n_init, max_iter=-101, tol=ModK._tol, GEV_=ModK._GEV_
    )
    with pytest.raises(ValueError, match="max iteration must be a positive"):
        _prepare_kwargs("ModKMeans", kwargs)

    kwargs = dict(
        n_init=ModK._n_init, max_iter=ModK.max_iter, tol=-101, GEV_=ModK._GEV_
    )
    with pytest.raises(
        ValueError, match="tolerance must be a positive number"
    ):
        _prepare_kwargs("ModKMeans", kwargs)

    kwargs = dict(
        n_init=ModK.n_init, max_iter=ModK.max_iter, tol=ModK.tol, GEV_=101
    )
    with pytest.raises(
        ValueError, match="'GEV_' should be a percentage between 0 and 1"
    ):
        _prepare_kwargs("ModKMeans", kwargs)


def test_invalid_read(tmp_path):
    """Test invalid arguments provided to read."""
    with pytest.raises(TypeError, match="'fname' must be an instance of"):
        _read_cluster(101)

    fname = directory / "sample_audvis_trunc_raw.fif"
    with pytest.raises(
        RuntimeError, match="Could not find clustering solution data."
    ):
        _read_cluster(fname)

    # save an ICA
    ica = ICA(n_components=5, method="infomax")
    ica.fit(raw, picks="eeg")
    ica.save(tmp_path / "decomposition-ica.fif")
    # try loading the ICA
    with pytest.raises(
        RuntimeError, match="Could not find clustering solution data."
    ):
        _read_cluster(fname)
