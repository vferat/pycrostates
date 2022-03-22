"""Test _imports.py"""

import pytest

from mne import create_info
from mne.io.constants import FIFF
import numpy as np

from pycrostates.io import ChInfo
from pycrostates.utils._logs import logger, set_log_level

set_log_level('INFO')
logger.propagate = True


def test_create_from_info():
    """Test creation of a ChInfo from an Info instance."""
    info = create_info(ch_names=['1', '2', '3'], sfreq=1, ch_types='eeg')
    chinfo = ChInfo(info=info)

    # basic tests
    assert chinfo['bads'] == []
    assert chinfo['ch_names'] == ['1', '2', '3']
    for k, ch in enumerate(chinfo['chs']):
        assert ch['kind'] == FIFF.FIFFV_EEG_CH
        assert ch['coil_type'] == FIFF.FIFFV_COIL_EEG
        assert ch['unit'] == FIFF.FIFF_UNIT_V
        assert ch['coord_frame'] == FIFF.FIFFV_COORD_HEAD
        assert ch['ch_name'] == str(k+1)
        assert all(np.isnan(elt) for elt in ch['loc'])
    assert chinfo['dig'] is None
    assert chinfo['custom_ref_applied'] == FIFF.FIFFV_MNE_CUSTOM_REF_OFF
    assert chinfo['nchan'] == 3

    # test changing a value from info
    with info._unlock():
        info['custom_ref_applied'] = FIFF.FIFFV_MNE_CUSTOM_REF_ON
    assert chinfo['custom_ref_applied'] == FIFF.FIFFV_MNE_CUSTOM_REF_OFF

    # test with info that has montage
    info = create_info(ch_names=['Fp1', 'Fp2', 'Fpz'], sfreq=1, ch_types='eeg')
    info.set_montage('standard_1020')
    chinfo = ChInfo(info=info)
    assert chinfo['ch_names'] == ['Fp1', 'Fp2', 'Fpz']
    for k, ch in enumerate(info['chs']):
        assert ch['coord_frame'] == FIFF.FIFFV_COORD_HEAD
        assert not all(np.isnan(elt) for elt in ch['loc'])
    assert chinfo['dig'] is not None
    assert chinfo['nchan'] == 3

    # test with multiple channel types
    ch_names = [f'MEG{n:03}' for n in range(1, 10)] + ['EOG001']
    ch_types = ['mag', 'grad', 'grad'] * 3 + ['eog']
    kinds_ = {
        'mag': FIFF.FIFFV_MEG_CH,
        'grad': FIFF.FIFFV_MEG_CH,
        'eog': FIFF.FIFFV_EOG_CH,
        }
    coil_type_ = {
        'mag': FIFF.FIFFV_COIL_VV_MAG_T3,
        'grad': FIFF.FIFFV_COIL_VV_PLANAR_T1,
        'eog': FIFF.FIFFV_COIL_NONE,
        }
    unit_ = {
        'mag': FIFF.FIFF_UNIT_T,
        'grad': FIFF.FIFF_UNIT_T_M,
        'eog': FIFF.FIFF_UNIT_V,
        }
    info = create_info(ch_names, ch_types=ch_types, sfreq=1)
    chinfo = ChInfo(info=info)
    assert chinfo['ch_names'] == ch_names
    for k, ch in enumerate(chinfo['chs']):
        assert ch['kind'] == kinds_[ch_types[k]]
        assert ch['coil_type'] == coil_type_[ch_types[k]]
        assert ch['unit'] == unit_[ch_types[k]]
        if 'EOG' in ch['ch_name']:
            assert ch['coord_frame'] == FIFF.FIFFV_COORD_UNKNOWN
        else:
            assert ch['coord_frame'] == FIFF.FIFFV_COORD_DEVICE
        assert ch['ch_name'] == ch_names[k]
        assert all(np.isnan(elt) for elt in ch['loc'])


def test_create_from_info_invalid_arguments():
    """Test creation of a ChInfo from an Info instance with invalid args."""
    ch_names = ['1', '2', '3']
    ch_types = ['eeg'] * 3
    info = create_info(ch_names=ch_names, sfreq=1, ch_types=ch_types)

    with pytest.raises(RuntimeError, match="If 'info' is provided"):
        ChInfo(info, ch_names=ch_names)
    with pytest.raises(RuntimeError, match="If 'info' is provided"):
        ChInfo(info, ch_types=ch_types)
    with pytest.raises(TypeError, match="'info' must be an instance of None "
                       "or Info"):
        ChInfo(info=ch_names)


def test_create_from_channels():
    """Test creation of a ChInfo from channel names and types."""
    ch_names = ['1', '2', '3']
    ch_types = ['eeg'] * 3
    chinfo = ChInfo(ch_names=ch_names, ch_types=ch_types)

    # basic tests
    assert chinfo['bads'] == []
    assert chinfo['ch_names'] == ['1', '2', '3']
    for k, ch in enumerate(chinfo['chs']):
        assert ch['kind'] == FIFF.FIFFV_EEG_CH
        assert ch['coil_type'] == FIFF.FIFFV_COIL_EEG
        assert ch['unit'] == FIFF.FIFF_UNIT_V
        assert ch['coord_frame'] == FIFF.FIFFV_COORD_HEAD
        assert ch['ch_name'] == str(k+1)
        assert all(np.isnan(elt) for elt in ch['loc'])
    assert chinfo['dig'] is None
    assert chinfo['custom_ref_applied'] == FIFF.FIFFV_MNE_CUSTOM_REF_OFF
    assert chinfo['nchan'] == 3

    # test with multiple channel types
    ch_names = [f'MEG{n:03}' for n in range(1, 10)] + ['EOG001']
    ch_types = ['mag', 'grad', 'grad'] * 3 + ['eog']
    kinds_ = {
        'mag': FIFF.FIFFV_MEG_CH,
        'grad': FIFF.FIFFV_MEG_CH,
        'eog': FIFF.FIFFV_EOG_CH,
        }
    coil_type_ = {
        'mag': FIFF.FIFFV_COIL_VV_MAG_T3,
        'grad': FIFF.FIFFV_COIL_VV_PLANAR_T1,
        'eog': FIFF.FIFFV_COIL_NONE,
        }
    unit_ = {
        'mag': FIFF.FIFF_UNIT_T,
        'grad': FIFF.FIFF_UNIT_T_M,
        'eog': FIFF.FIFF_UNIT_V,
        }
    chinfo = ChInfo(ch_names=ch_names, ch_types=ch_types)
    assert chinfo['ch_names'] == ch_names
    for k, ch in enumerate(chinfo['chs']):
        assert ch['kind'] == kinds_[ch_types[k]]
        assert ch['coil_type'] == coil_type_[ch_types[k]]
        assert ch['unit'] == unit_[ch_types[k]]
        if 'EOG' in ch['ch_name']:
            assert ch['coord_frame'] == FIFF.FIFFV_COORD_UNKNOWN
        else:
            assert ch['coord_frame'] == FIFF.FIFFV_COORD_DEVICE
        assert ch['ch_name'] == ch_names[k]
        assert all(np.isnan(elt) for elt in ch['loc'])


def test_Create_from_channels_invalid_arguments():
    """Test creation of a ChInfo from channel names and types with invalid
    args."""
    ch_names = ['1', '2', '3']
    ch_types = ['eeg'] * 3
    info = create_info(ch_names=ch_names, sfreq=1, ch_types=ch_types)

    with pytest.raises(RuntimeError, match="If 'ch_names' and 'ch_types'"):
        ChInfo(info=info, ch_names=ch_names, ch_types=ch_types)
    with pytest.raises(RuntimeError, match="If 'ch_names' and 'ch_types'"):
        ChInfo(info=None, ch_names=ch_names, ch_types=None)
    with pytest.raises(RuntimeError, match="If 'ch_names' and 'ch_types'"):
        ChInfo(info=None, ch_names=None, ch_types=ch_types)
    with pytest.raises(TypeError, match="'ch_names' must be an instance of"):
        ChInfo(ch_names=dict(), ch_types=ch_types)
    with pytest.raises(TypeError, match="'ch_types' must be an instance of"):
        ChInfo(ch_names=ch_names, ch_types=5)


def test_create_without_arguments():
    """Test error raised if both arguments are None."""
    with pytest.raises(RuntimeError, match="Either 'info' or 'ch_names' and "
                       "'ch_types' must not be None."):
        ChInfo()


def test_montage():
    pass


def test_contains():
    pass


def test_copy():
    pass


def test_repr():
    pass
