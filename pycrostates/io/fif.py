"""Contains I/O operation form and towards .fif format (MEGIN / MNE)."""

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
    write_double_matrix, write_dig_points, write_name_list)
from mne.io._digitization import _read_dig_fif, _format_dig_points

from . import ChInfo
from ..utils._logs import logger


# ----------------------------------------------------------------------------
def write_cluster(fname, cluster_centers, chinfo):
    """Save clustering to disk."""
    logger.info('Writing clustering solution to %s...', fname)
    with start_and_end_file(fname) as fid:
        # write info
        start_block(fid, FIFF.FIFFB_MEAS)
        write_id(fid, FIFF.FIFF_BLOCK_ID)
        _write_meas_info(fid, chinfo)
        end_block(fid, FIFF.FIFFB_MEAS)

        # write data
        start_block(fid, FIFF.FIFFB_MNE_ICA)
        write_double_matrix(fid, FIFF.FIFF_MNE_ICA_MATRIX, cluster_centers)
        end_block(fid, FIFF.FIFFB_MNE_ICA)


def read_cluster(fname):
    """Read clustering from disk."""
    logger.info('Reading clustering solution from %s...', fname)
    fid, tree, _ = fiff_open(fname)
    info, _ = _read_meas_info(fid, tree, clean_bads=True)


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
