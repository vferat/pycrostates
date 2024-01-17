from pathlib import Path as Path
from typing import Union

from mne import Info
from numpy.typing import NDArray

from .._typing import CHInfo as CHInfo
from .._version import __version__ as __version__
from ..cluster import AAHCluster as AAHCluster
from ..cluster import ModKMeans as ModKMeans
from ..utils._checks import _check_type as _check_type
from ..utils._checks import _check_value as _check_value
from ..utils._docs import fill_doc as fill_doc
from ..utils._logs import logger as logger

def _write_cluster(
    fname: Union[str, Path],
    cluster_centers_: NDArray[float],
    chinfo: Union[CHInfo, Info],
    algorithm: str,
    cluster_names: list[str],
    fitted_data: NDArray[float],
    labels_: NDArray[int],
    **kwargs,
):
    """Save clustering solution to disk.

    Parameters
    ----------
    fname : str | Path
        Path to the ``.fif`` file where the clustering solution is saved.
    cluster_centers_ : array
        Cluster centers as a numpy array of shape (n_clusters, n_channels).
    chinfo : ChInfo
        Channel information (name, type, montage, ..)
    algorithm : str
        Clustering algorithm used. Valids are:
            'ModKMeans'
    cluster_names : list
        List of names for each of the clusters.
    fitted_data : array
        Data array used for fitting of shape (n_channels, n_samples)
    labels_ : array
        Array of labels for each sample of shape (n_samples, )
    """

def _prepare_kwargs(algorithm: str, kwargs: dict):
    """Prepare params to save from kwargs."""

def _read_cluster(fname: Union[str, Path]):
    """Read clustering solution from disk.

    Parameters
    ----------
    fname : str | Path
        Path to the ``.fif`` file where the clustering solution is saved.

    Returns
    -------
    cluster : _BaseCluster
        Loaded cluster solution.
    version : str
        pycrostates version used to save the cluster solution.
    """

def _check_fit_parameters_and_variables(fit_parameters: dict, fit_variables: dict):
    """Check that we have all the keys we are looking for and return algo."""

def _create_ModKMeans(
    cluster_centers_: NDArray[float],
    info: CHInfo,
    cluster_names: list[str],
    fitted_data: NDArray[float],
    labels_: NDArray[int],
    n_init: int,
    max_iter: int,
    tol: Union[int, float],
    GEV_: float,
):
    """Create a ModKMeans cluster."""

def _create_AAHCluster(
    cluster_centers_: NDArray[float],
    info: CHInfo,
    cluster_names: list[str],
    fitted_data: NDArray[float],
    labels_: NDArray[int],
    ignore_polarity: bool,
    normalize_input: bool,
    GEV_: float,
):
    """Create a AAHCluster object."""

def _write_meas_info(fid, info: CHInfo):
    """Write measurement info into a file id (from a fif file).

    Parameters
    ----------
    fid : file
        Open file descriptor.
    info : ChInfo
        Channel information.
    """

def _read_meas_info(fid, tree):
    """Read the measurement info.

    Parameters
    ----------
    fid : file
        Open file descriptor.
    tree : tree
        FIF tree structure.

    Returns
    -------
    info : ChInfo
        Channel information instance.
    """

def _serialize(dict_: dict, outer_sep: str = ";", inner_sep: str = ":"):
    """Aux function."""

def _deserialize(str_: str, outer_sep: str = ";", inner_sep: str = ":"):
    """Aux Function."""
