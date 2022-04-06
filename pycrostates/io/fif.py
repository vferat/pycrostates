"""Contains I/O operation form and towards .fif format (MEGIN / MNE)."""

import json
from numbers import Integral

from mne.io import Info
from mne.io.constants import FIFF
from mne.io.ctf_comp import write_ctf_comp, _read_ctf_comp
from mne.io.meas_info import _write_ch_infos, _read_bad_channels
from mne.io.open import fiff_open
from mne.io.proj import _write_proj, _read_proj
from mne.io.tag import _rename_list, read_tag
from mne.io.tree import dir_tree_find
from mne.io.write import (
    start_and_end_file, start_block, end_block, write_id, write_int,
    write_double_matrix, write_dig_points, write_name_list, write_string)
from mne.io._digitization import _read_dig_fif, _format_dig_points
import numpy as np

from . import ChInfo
from ..cluster import ModKMeans
from ..utils._checks import _check_value, _check_type
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
FIFF_MNE_ICA_INTERFACE_PARAMS -> algorithm + parameters
FIFF_MNE_ICA_MISC_PARAMS -> fitted variables (ending with '_')
"""


def write_cluster(fname, cluster_centers, chinfo, algorithm, **kwargs):
    """Save clustering to disk."""
    logger.info('Writing clustering solution to %s...', fname)

    # retrieve information to store from kwargs
    cluster_names, fitted_data, labels_, parameters, fitted_parameters = \
        _prepare_kwargs(kwargs)

    # algorithm
    _check_value(algorithm, ('modKmeans', ), 'algorithm')
    parameters['algorithm'] = algorithm

    with start_and_end_file(fname) as fid:
        # write info
        start_block(fid, FIFF.FIFFB_MEAS)
        write_id(fid, FIFF.FIFF_BLOCK_ID)
        _write_meas_info(fid, chinfo)
        end_block(fid, FIFF.FIFFB_MEAS)

        # start writing block
        start_block(fid, FIFF.FIFFB_MNE_ICA)

        # cluster-centers
        write_double_matrix(fid, FIFF.FIFF_MNE_ICA_MATRIX, cluster_centers)

        # write parameters
        write_string(fid, FIFF.FIFF_MNE_ICA_INTERFACE_PARAMS,
                     _serialize(parameters))

        # write fitted parameters
        write_string(fid, FIFF.FIFF_MNE_ICA_MISC_PARAMS,
                     _serialize(fitted_parameters))

        # write cluster_names
        if cluster_names is not None:
            write_name_list(fid, FIFF.FIFF_MNE_ROW_NAMES, cluster_names)

        # write fitted_data
        if fitted_data is not None:
            write_double_matrix(fid, FIFF.FIFF_MNE_ICA_WHITENER, fitted_data)

        # write labels
        if labels_ is not None:
            write_double_matrix(fid, FIFF.FIFF_MNE_ICA_PCA_MEAN, labels_)

        # close writing block
        end_block(fid, FIFF.FIFFB_MNE_ICA)


def _prepare_kwargs(kwargs: dict):
    """Prepare params to save from kwargs."""
    # defaults
    cluster_names = None
    fitted_data = None
    labels_ = None

    base_kwargs = ['cluster_names', 'fitted_data', 'labels_']
    modkmeans_kwargs_params = ['n_init', 'max_iter', 'tol']
    modkmeans_kwargs_fitted_params = ['GEV_']

    parameters = {key: None for key in modkmeans_kwargs_params}
    fitted_parameters = {key: None for key in modkmeans_kwargs_fitted_params}

    for key, value in kwargs.items():
        if key not in base_kwargs + list(parameters) + list(fitted_parameters):
            continue

        # base
        elif key == 'cluster_names':
            _check_type(value, (list, ), 'cluster_names')
            cluster_names = value
        elif key == 'fitted_data':
            _check_type(value, (np.ndarray, ), 'fitted_data')
            if value.ndim != 2:
                raise ValueError("Fitted data should be a 2D array.")
            fitted_data = value.astype(np.float64)
        elif key == 'labels_':
            _check_type(value, (np.ndarray, ), 'labels_')
            if value.ndim != 1:
                raise ValueError('Labels data should be a 1D array.')
            labels_ = value.reshape(-1, 1).astype(np.float64)

        # ModKMeans
        elif key == 'n_init':
            parameters['n_init'] = ModKMeans._check_n_init(value)
        elif key == 'max_iter':
            parameters['max_iter'] = ModKMeans._check_max_iter(value)
        elif key == 'tol':
            parameters['tol'] = ModKMeans._check_tol(value)
        elif key == 'GEV_':
            _check_type(value, ('numeric', ), 'GEV_')
            if value < 0 or 1 < value:
                raise ValueError('GEV should be a percentage between 0 and 1.')
            fitted_parameters['GEV_'] = value

    return cluster_names, fitted_data, labels_, parameters, fitted_parameters


def read_cluster(fname):
    """Read clustering from disk."""
    logger.info('Reading clustering solution from %s...', fname)
    fid, tree, _ = fiff_open(fname)
    info = _read_meas_info(fid, tree)
    data_tree = dir_tree_find(tree, FIFF.FIFFB_MNE_ICA)
    if len(data_tree) == 0:
        fid.close()
        raise ValueError('Could not find clustering solution data.')

    # init variables to search
    parameters = dict()
    fitted_parameters = dict()
    cluster_names = None
    fitted_data = None
    labels_ = None

    data_tree = data_tree[0]
    for data in data_tree['directory']:
        kind = data.kind
        pos = data.pos
        # cluster_centers
        if kind == FIFF.FIFF_MNE_ICA_MATRIX:
            tag = read_tag(fid, pos)
            cluster_centers = tag.data.astype(np.float64)
        # parameters
        elif kind == FIFF.FIFF_MNE_ICA_INTERFACE_PARAMS:
            tag = read_tag(fid, pos)
            parameters = _deserialize(tag.data)
        # fitted_parameters
        elif kind == FIFF.FIFF_MNE_ICA_MISC_PARAMS:
            tag = read_tag(fid, pos)
            fitted_parameters = _deserialize(tag.data)
        # cluster_names
        elif kind == FIFF.FIFF_MNE_ROW_NAMES:
            tag = read_tag(fid, pos)
            cluster_names = tag.data.split(':')
        # fitted_data
        elif kind == FIFF.FIFF_MNE_ICA_WHITENER:
            tag = read_tag(fid, pos)
            fitted_data = tag.data.astype(np.float64)
        # labels
        elif kind == FIFF.FIFF_MNE_ICA_PCA_MEAN:
            tag = read_tag(fid, pos)
            labels_ = tag.data[:, 0].astype(np.int64)

    fid.close()

    # make sure we have all the information required
    # TODO

    # reconstruct clustering instance
    if parameters['algorithm'] == 'modKmeans':
        inst = ModKMeans(
            cluster_centers.shape[0],
            n_init=parameters['n_init'],
            max_iter=parameters['max_iter'],
            tol=parameters['tol'],
            random_state=None,
            )
        inst._cluster_centers_ = cluster_centers
        inst._info = info
        inst._cluster_names = cluster_names
        inst._fitted_data = fitted_data
        inst._labels_ = labels_
        inst._GEV_ = fitted_parameters['GEV_']
        inst._fitted = True
    else:
        raise ValueError(
            f"Algorithm '{parameters['algorithm']}' is not supported.")

    return inst


# ----------------------------------------------------------------------------
def _write_meas_info(fid, info, data_type=None, reset_range=True):
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
    write_dig_points(fid, info['dig'], block=True)

    # Projectors
    _write_proj(fid, info['projs'], ch_names_mapping={})

    # Bad channels
    if len(info['bads']) > 0:
        bads = _rename_list(info['bads'], ch_names_mapping={})
        start_block(fid, FIFF.FIFFB_MNE_BAD_CHANNELS)
        write_name_list(fid, FIFF.FIFF_MNE_CH_NAME_LIST, bads)
        end_block(fid, FIFF.FIFFB_MNE_BAD_CHANNELS)

    # General
    write_int(fid, FIFF.FIFF_NCHAN, info['nchan'])
    if info.get('custom_ref_applied'):
        write_int(fid, FIFF.FIFF_MNE_CUSTOM_REF, info['custom_ref_applied'])

    # Channel information
    _write_ch_infos(fid, info['chs'], reset_range=True, ch_names_mapping={})

    # CTF compensation info
    comps = info['comps']
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
    """
    # Find the desired blocks
    meas = dir_tree_find(tree, FIFF.FIFFB_MEAS)
    if len(meas) == 0:
        raise ValueError('Could not find measurement data')
    if len(meas) > 1:
        raise ValueError('Cannot read more that 1 measurement data')
    meas = meas[0]

    meas_info = dir_tree_find(meas, FIFF.FIFFB_MEAS_INFO)
    if len(meas_info) == 0:
        raise ValueError('Could not find measurement info')
    if len(meas_info) > 1:
        raise ValueError('Cannot read more that 1 measurement info')
    meas_info = meas_info[0]

    # Read measurement info
    nchan = None
    chs = []
    custom_ref_applied = FIFF.FIFFV_MNE_CUSTOM_REF_OFF
    for k in range(meas_info['nent']):
        kind = meas_info['directory'][k].kind
        pos = meas_info['directory'][k].pos
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
        raise ValueError('Number of channels is not defined')
    if len(chs) == 0:
        raise ValueError('Channel information not defined')
    if len(chs) != nchan:
        raise ValueError('Incorrect number of channel definitions found')

    # Locate the Polhemus data
    dig = _read_dig_fif(fid, meas_info)

    # Load the SSP data
    projs = _read_proj(
        fid, meas_info, ch_names_mapping=None)

    # Load the CTF compensation data
    comps = _read_ctf_comp(fid, meas_info, chs, ch_names_mapping=None)

    # Load the bad channel list
    bads = _read_bad_channels(fid, meas_info, ch_names_mapping=None)

    # Put the data together
    info = Info(file_id=tree['id'])
    info._unlocked = True
    info['chs'] = chs
    info['dig'] = _format_dig_points(dig)
    info['bads'] = bads
    info._update_redundant()
    info['bads'] = [b for b in bads if b in info['ch_names']]  # sanity-check
    info['projs'] = projs
    info['comps'] = comps
    info['custom_ref_applied'] = custom_ref_applied
    info._check_consistency()
    info._unlocked = False

    return ChInfo(info)


# ----------------------------------------------------------------------------
def _serialize(dict_, outer_sep=';', inner_sep=':'):
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


def _deserialize(str_, outer_sep=';', inner_sep=':'):
    """Aux Function."""
    out = {}
    for mapping in str_.split(outer_sep):
        k, v = mapping.split(inner_sep, 1)
        out[k] = json.loads(v)
    return out
