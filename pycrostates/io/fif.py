"""Contains I/O operation form and towards .fif format (MEGIN / MNE)."""

from copy import deepcopy

from mne.io.constants import FIFF
from mne.io.ctf_comp import write_ctf_comp
from mne.io.meas_info import (
    _make_ch_names_mapping, _write_ch_infos, _rename_comps)
from mne.io.proj import _write_proj
from mne.io.tag import _rename_list
from mne.io.write import (
    start_and_end_file, start_block, end_block, write_id, write_int,
    write_double_matrix, write_dig_points, write_name_list)

from ..utils._logs import logger


def write_cluster(fname, cluster_centers, chinfo):
    """Save clustering to disk."""
    logger.info('Writing clustering solution to %s...', fname)
    with start_and_end_file(fname) as fid:
        # write info
        start_block(fid, FIFF.FIFFB_MEAS)
        write_id(fid, FIFF.FIFF_BLOCK_ID)
        write_meas_info(fid, chinfo)
        end_block(fid, FIFF.FIFFB_MEAS)

        # write data
        start_block(fid, FIFF.FIFFB_MNE_ICA)
        write_double_matrix(fid, FIFF.FIFF_MNE_ICA_MATRIX, cluster_centers)
        end_block(fid, FIFF.FIFFB_MNE_ICA)


def read_cluster(fname):
    """Read clustering from disk."""
    pass


def write_meas_info(fid, info, data_type=None, reset_range=True):
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
    ch_names_mapping = _make_ch_names_mapping(info['chs'])
    _write_proj(fid, info['projs'], ch_names_mapping=ch_names_mapping)

    # Bad channels
    if len(info['bads']) > 0:
        bads = _rename_list(info['bads'], ch_names_mapping)
        start_block(fid, FIFF.FIFFB_MNE_BAD_CHANNELS)
        write_name_list(fid, FIFF.FIFF_MNE_CH_NAME_LIST, bads)
        end_block(fid, FIFF.FIFFB_MNE_BAD_CHANNELS)

    # General
    write_int(fid, FIFF.FIFF_NCHAN, info['nchan'])
    if info.get('custom_ref_applied'):
        write_int(fid, FIFF.FIFF_MNE_CUSTOM_REF, info['custom_ref_applied'])

    # Channel information
    _write_ch_infos(fid, info['chs'], reset_range=True,
                    ch_names_mapping=ch_names_mapping)

    # CTF compensation info
    comps = info['comps']
    if ch_names_mapping:
        comps = deepcopy(comps)
        _rename_comps(comps, ch_names_mapping)
    write_ctf_comp(fid, comps)
