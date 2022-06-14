"""Contains I/O operation from and towards .fif format (MEGIN / MNE)."""

import json
import operator
from functools import reduce
from numbers import Integral
from pathlib import Path
from typing import List, Union

import numpy as np
from mne.io import Info
from mne.io._digitization import _format_dig_points, _read_dig_fif
from mne.io.constants import FIFF
from mne.io.ctf_comp import _read_ctf_comp, write_ctf_comp
from mne.io.meas_info import _read_bad_channels, _write_ch_infos
from mne.io.open import fiff_open
from mne.io.proj import _read_proj, _write_proj
from mne.io.tag import read_tag
from mne.io.tree import dir_tree_find
from mne.io.write import (
    end_block,
    start_and_end_file,
    start_block,
    write_dig_points,
    write_double_matrix,
    write_id,
    write_int,
    write_name_list,
    write_string,
)
from numpy.typing import NDArray

from .. import __version__
from .._typing import CHInfo
from ..cluster import ModKMeans
from ..utils._checks import _check_type, _check_value
from ..utils._docs import fill_doc
from ..utils._logs import logger

# ----------------------------------------------------------------------------
"""
To store a clustering solution, the FIFF tags for ICA are used.
From the FIFF specification:

    #
    # 3601... values associated with ICA decomposition
    #


    mne_ica_interface_params    3601 ? "ICA interface parameters"
    mne_ica_channel_names       3602 ? "ICA channel names"
    mne_ica_whitener            3603 ? "ICA whitener"
    mne_ica_pca_components      3604 ? "PCA components"
    mne_ica_pca_explained_var   3605 ? "PCA explained variance"
    mne_ica_pca_mean            3606 ? "PCA mean"
    mne_ica_matrix              3607 ? "ICA unmixing matrix"
    mne_ica_bads                3608 ? "ICA bad sources"
    mne_ica_misc_params         3609 ? "ICA misc params"


FIFF_MNE_ICA_MATRIX -> cluster_centers_
FIFF_MNE_ROW_NAMES -> cluster_names
FIFF_MNE_ICA_WHITENER -> fitted_data
FIFF_MNE_ICA_PCA_MEAN -> labels
FIFF_MNE_ICA_INTERFACE_PARAMS -> algorithm + version + fit parameters
FIFF_MNE_ICA_MISC_PARAMS -> fit variables (ending with '_')
"""


@fill_doc
def _write_cluster(
    fname: Union[str, Path],
    cluster_centers_: NDArray[float],
    chinfo: Union[CHInfo, Info],
    algorithm: str,
    cluster_names: List[str],
    fitted_data: NDArray[float],
    labels_: NDArray[int],
    **kwargs,
):
    """Save clustering solution to disk.

    Parameters
    ----------
    %(fname_fiff)s
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
    from . import ChInfo

    # error checking on input
    _check_type(fname, ("path-like",), "fname")
    _check_type(cluster_centers_, (np.ndarray,), "cluster_centers_")
    if cluster_centers_.ndim != 2:
        raise ValueError("Argument 'cluster_centers_' should be a 2D array.")
    _check_type(chinfo, (Info, ChInfo), "chinfo")
    if isinstance(chinfo, Info):
        chinfo = ChInfo(chinfo)  # convert to ChInfo if a MNE Info is provided
    _check_type(algorithm, (str,), "algorithm")
    _check_value(algorithm, ("ModKMeans",), "algorithm")
    _check_type(cluster_names, (list,), "cluster_names")
    if len(cluster_names) != cluster_centers_.shape[0]:
        raise ValueError(
            "Argument 'cluster_names' and 'cluster_centers_' shapes do not "
            "match."
        )
    _check_type(fitted_data, (np.ndarray,), "fitted_data")
    if fitted_data.ndim != 2:
        raise ValueError("Argument 'fitted_data' should be a 2D array.")
    _check_type(labels_, (np.ndarray,), "labels_")
    if labels_.ndim != 1:
        raise ValueError("Argument 'labels_' should be a 1D array.")

    # logging
    logger.info("Writing clustering solution to %s...", fname)

    # retrieve information to store from kwargs
    fit_parameters, fit_variables = _prepare_kwargs(algorithm, kwargs)

    with start_and_end_file(fname) as fid:
        # write channel info
        start_block(fid, FIFF.FIFFB_MEAS)
        write_id(fid, FIFF.FIFF_BLOCK_ID)
        _write_meas_info(fid, chinfo)
        end_block(fid, FIFF.FIFFB_MEAS)

        # start writing block
        start_block(fid, FIFF.FIFFB_MNE_ICA)

        # ------------------------------------------------------------
        # cluster_centers_
        write_double_matrix(
            fid, FIFF.FIFF_MNE_ICA_MATRIX, cluster_centers_.astype(np.float64)
        )
        # write cluster_names
        write_name_list(fid, FIFF.FIFF_MNE_ROW_NAMES, cluster_names)
        # write fitted_data
        write_double_matrix(
            fid, FIFF.FIFF_MNE_ICA_WHITENER, fitted_data.astype(np.float64)
        )
        # write labels_
        write_double_matrix(
            fid,
            FIFF.FIFF_MNE_ICA_PCA_MEAN,
            labels_.reshape(-1, 1).astype(np.float64),
        )
        # write fit_parameters
        write_string(
            fid, FIFF.FIFF_MNE_ICA_INTERFACE_PARAMS, _serialize(fit_parameters)
        )
        # write fit_variables
        write_string(
            fid, FIFF.FIFF_MNE_ICA_MISC_PARAMS, _serialize(fit_variables)
        )
        # ------------------------------------------------------------

        # close writing block
        end_block(fid, FIFF.FIFFB_MNE_ICA)


def _prepare_kwargs(algorithm: str, kwargs: dict):
    """Prepare params to save from kwargs."""
    valids = {
        "ModKMeans": {
            "parameters": ["n_init", "max_iter", "tol"],
            "variables": ["GEV_"],
        },
    }

    # retrieve list of expected kwargs for this algorithm
    expected = set(reduce(operator.concat, valids[algorithm].values()))

    # check that we do have a value provided for each expected key
    keys = set(key for key in kwargs if kwargs[key] is not None)
    conditions = (
        len(keys.difference(expected)) != 0,
        len(expected.difference(keys)) != 0,
    )
    if any(conditions):
        raise ValueError(
            f"Wrong kwargs provided for algorithm '{algorithm}'. Expected: "
            f"{', '.join(expected)} should not be None."
        )

    fit_parameters = dict(algorithm=algorithm, version=__version__)
    fit_variables = {}
    for key, value in kwargs.items():
        if key not in expected:
            continue

        # ModKMeans
        if key == "n_init":
            fit_parameters["n_init"] = ModKMeans._check_n_init(value)
        elif key == "max_iter":
            fit_parameters["max_iter"] = ModKMeans._check_max_iter(value)
        elif key == "tol":
            fit_parameters["tol"] = ModKMeans._check_tol(value)
        elif key == "GEV_":
            _check_type(value, ("numeric",), "GEV_")
            if value < 0 or 1 < value:
                raise ValueError(
                    "Argument 'GEV_' should be a percentage between 0 and 1. "
                    f"Provided: '{value}'."
                )
            fit_variables["GEV_"] = value

    return fit_parameters, fit_variables


@fill_doc
def _read_cluster(fname: Union[str, Path]):
    """Read clustering solution from disk.

    Parameters
    ----------
    %(fname_fiff)s

    Returns
    -------
    cluster : _BaseCluster
        Loaded cluster solution.
    version : str
        pycrostates version used to save the cluster solution.
    """
    # error checking on input
    _check_type(fname, ("path-like",), "fname")

    # logging
    logger.info("Reading clustering solution from %s...", fname)

    # open file
    fid, tree, _ = fiff_open(fname)
    info = _read_meas_info(fid, tree)
    data_tree = dir_tree_find(tree, FIFF.FIFFB_MNE_ICA)
    if len(data_tree) == 0:
        fid.close()
        raise RuntimeError("Could not find clustering solution data.")

    # init variables to search
    cluster_centers_ = None
    cluster_names = None
    fitted_data = None
    labels_ = None
    fit_parameters = None
    fit_variables = None

    try:
        data_tree = data_tree[0]
        for data in data_tree["directory"]:
            kind = data.kind
            pos = data.pos
            # cluster_centers_
            if kind == FIFF.FIFF_MNE_ICA_MATRIX:
                tag = read_tag(fid, pos)
                cluster_centers_ = tag.data.astype(np.float64)
            # cluster_names
            elif kind == FIFF.FIFF_MNE_ROW_NAMES:
                tag = read_tag(fid, pos)
                cluster_names = tag.data.split(":")
            # fitted_data
            elif kind == FIFF.FIFF_MNE_ICA_WHITENER:
                tag = read_tag(fid, pos)
                fitted_data = tag.data.astype(np.float64)
            # labels
            elif kind == FIFF.FIFF_MNE_ICA_PCA_MEAN:
                tag = read_tag(fid, pos)
                labels_ = tag.data[:, 0].astype(np.int64)
            # fit_parameters
            elif kind == FIFF.FIFF_MNE_ICA_INTERFACE_PARAMS:
                tag = read_tag(fid, pos)
                fit_parameters = _deserialize(tag.data)
            # fit_variables
            elif kind == FIFF.FIFF_MNE_ICA_MISC_PARAMS:
                tag = read_tag(fid, pos)
                fit_variables = _deserialize(tag.data)
    except Exception:
        raise RuntimeError("Could not find clustering solution data.")
    finally:
        fid.close()

    # re-group variables and make sure we have all the information required
    data = (
        cluster_centers_,
        info,
        cluster_names,
        fitted_data,
        labels_,
        fit_parameters,
        fit_variables,
    )
    if any(elt is None for elt in data):
        raise RuntimeError(
            "One of the required tag was not found in .fif file."
        )
    algorithm, version = _check_fit_parameters_and_variables(
        fit_parameters, fit_variables
    )

    # reconstruct cluster instance
    function = {"ModKMeans": _create_ModKMeans}

    return (
        function[algorithm](
            cluster_centers_,
            info,
            cluster_names,
            fitted_data,
            labels_,
            **fit_parameters,
            **fit_variables,
        ),
        version,
    )


def _check_fit_parameters_and_variables(
    fit_parameters: dict,
    fit_variables: dict,
):
    """Check that we have all the keys we are looking for and return algo."""
    valids = {
        "ModKMeans": {
            "parameters": ["n_init", "max_iter", "tol"],
            "variables": ["GEV_"],
        },
    }
    if "algorithm" not in fit_parameters:
        raise ValueError("Key 'algorithm' is missing from .fif file.")
    if "version" not in fit_parameters:
        raise ValueError("Key 'version' is missing from .fif file.")
    algorithm = fit_parameters["algorithm"]
    if algorithm not in valids:
        raise ValueError(f"Algorithm '{algorithm}' is not supported.")
    del fit_parameters["algorithm"]
    version = fit_parameters["version"]
    del fit_parameters["version"]
    expected = set(reduce(operator.concat, valids[algorithm].values()))
    diff = set(list(fit_parameters) + list(fit_variables)).difference(expected)
    if len(diff) != 0:
        raise RuntimeError("Unexpected parameters and variables in .fif file.")
    return algorithm, version


def _create_ModKMeans(
    cluster_centers_: NDArray[float],
    info: CHInfo,
    cluster_names: List[str],
    fitted_data: NDArray[float],
    labels_: NDArray[int],
    n_init: int,
    max_iter: int,
    tol: Union[int, float],
    GEV_: float,
):
    """Create a ModKMeans cluster."""
    cluster = ModKMeans(
        cluster_centers_.shape[0], n_init, max_iter, tol, random_state=None
    )
    cluster._cluster_centers_ = cluster_centers_
    cluster._info = info
    cluster._cluster_names = cluster_names
    cluster._fitted_data = fitted_data
    cluster._labels_ = labels_
    cluster._GEV_ = GEV_
    cluster._fitted = True
    return cluster


# ----------------------------------------------------------------------------
def _write_meas_info(fid, info: CHInfo):
    """Write measurement info into a file id (from a fif file).

    Parameters
    ----------
    fid : file
        Open file descriptor.
    info : ChInfo
        Channel information.
    """
    info._check_consistency()

    # Measurement info
    start_block(fid, FIFF.FIFFB_MEAS_INFO)

    # Polhemus data
    write_dig_points(fid, info["dig"], block=True)

    # Projectors
    _write_proj(fid, info["projs"], ch_names_mapping={})

    # Bad channels
    if len(info["bads"]) > 0:
        start_block(fid, FIFF.FIFFB_MNE_BAD_CHANNELS)
        write_name_list(fid, FIFF.FIFF_MNE_CH_NAME_LIST, info["bads"])
        end_block(fid, FIFF.FIFFB_MNE_BAD_CHANNELS)

    # General
    write_int(fid, FIFF.FIFF_NCHAN, info["nchan"])
    if info.get("custom_ref_applied"):
        write_int(fid, FIFF.FIFF_MNE_CUSTOM_REF, info["custom_ref_applied"])

    # Channel information
    _write_ch_infos(fid, info["chs"], reset_range=True, ch_names_mapping={})

    # CTF compensation info
    comps = info["comps"]
    write_ctf_comp(fid, comps)


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
    from . import ChInfo

    # Find the desired blocks
    meas = dir_tree_find(tree, FIFF.FIFFB_MEAS)
    if len(meas) == 0:
        raise ValueError("Could not find measurement data")
    if len(meas) > 1:
        raise ValueError("Cannot read more that 1 measurement data")
    meas = meas[0]

    meas_info = dir_tree_find(meas, FIFF.FIFFB_MEAS_INFO)
    if len(meas_info) == 0:
        raise ValueError("Could not find measurement info")
    if len(meas_info) > 1:
        raise ValueError("Cannot read more that 1 measurement info")
    meas_info = meas_info[0]

    # Read measurement info
    nchan = None
    chs = []
    custom_ref_applied = FIFF.FIFFV_MNE_CUSTOM_REF_OFF
    for k in range(meas_info["nent"]):
        kind = meas_info["directory"][k].kind
        pos = meas_info["directory"][k].pos
        if kind == FIFF.FIFF_NCHAN:
            tag = read_tag(fid, pos)
            nchan = int(tag.data)
        elif kind == FIFF.FIFF_CH_INFO:
            tag = read_tag(fid, pos)
            chs.append(tag.data)
        elif kind == FIFF.FIFF_MNE_CUSTOM_REF:
            tag = read_tag(fid, pos)
            custom_ref_applied = int(tag.data)

    # Check that we have everything we need
    if nchan is None:
        raise ValueError("Number of channels is not defined")
    if len(chs) == 0:
        raise ValueError("Channel information not defined")
    if len(chs) != nchan:
        raise ValueError("Incorrect number of channel definitions found")

    # Locate the Polhemus data
    dig = _read_dig_fif(fid, meas_info)

    # Load the SSP data
    projs = _read_proj(fid, meas_info, ch_names_mapping=None)

    # Load the CTF compensation data
    comps = _read_ctf_comp(fid, meas_info, chs, ch_names_mapping=None)

    # Load the bad channel list
    bads = _read_bad_channels(fid, meas_info, ch_names_mapping=None)

    # Put the data together
    info = Info(file_id=tree["id"])
    info._unlocked = True
    info["chs"] = chs
    info["dig"] = _format_dig_points(dig)
    info["bads"] = bads
    info._update_redundant()
    info["bads"] = [b for b in bads if b in info["ch_names"]]  # sanity-check
    info["projs"] = projs
    info["comps"] = comps
    info["custom_ref_applied"] = custom_ref_applied
    info._check_consistency()
    info._unlocked = False

    return ChInfo(info)


# ----------------------------------------------------------------------------
def _serialize(dict_: dict, outer_sep: str = ";", inner_sep: str = ":"):
    """Aux function."""
    s = []
    for key, value in dict_.items():
        if callable(value):
            value = value.__name__
        elif isinstance(value, Integral):
            value = int(value)
        elif isinstance(value, dict):
            # py35 json does not support numpy int64
            for subkey, subvalue in value.items():
                if isinstance(subvalue, list):
                    if len(subvalue) > 0:
                        if isinstance(subvalue[0], (int, np.integer)):
                            value[subkey] = [int(i) for i in subvalue]

        s.append(key + inner_sep + json.dumps(value))

    return outer_sep.join(s)


def _deserialize(str_: str, outer_sep: str = ";", inner_sep: str = ":"):
    """Aux Function."""
    out = {}
    for mapping in str_.split(outer_sep):
        k, v = mapping.split(inner_sep, 1)
        out[k] = json.loads(v)
    return out
