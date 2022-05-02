from mne import create_info
from mne.io.constants import FIFF
import pytest

from pycrostates.io import ChInfo
from pycrostates.utils import _compare_infos
from pycrostates.utils._logs import logger, set_log_level


set_log_level('INFO')
logger.propagate = True


def test_compare_infos(caplog):
    """Test _compare_infos(cluster_info, inst_info)."""
    # minimum chinfo
    chinfo = ChInfo(ch_names=['Fpz', 'Cz', 'CPz'], ch_types='eeg')
    _compare_infos(chinfo, chinfo.copy())

    # with montage
    chinfo.set_montage('standard_1020')
    caplog.clear()
    _compare_infos(chinfo, chinfo.copy())
    assert 'does not have the same channels montage' not in caplog.text

    # with MNE info without montage
    info = create_info(['Fpz', 'Cz', 'CPz'], 1, 'eeg')
    caplog.clear()
    _compare_infos(chinfo, info)
    assert 'does not have the same channels montage' in caplog.text
    caplog.clear()
    _compare_infos(info, chinfo)
    assert 'does not have the same channels montage' in caplog.text

    # with MNE info with montage
    info = create_info(['Fpz', 'Cz', 'CPz'], 1, 'eeg')
    info.set_montage('standard_1020')
    caplog.clear()
    _compare_infos(chinfo, info)
    assert 'does not have the same channels montage' not in caplog.text
    caplog.clear()
    _compare_infos(info, chinfo)
    assert 'does not have the same channels montage' not in caplog.text
    caplog.clear()

    # with different channels
    chinfo1 = ChInfo(ch_names=['Fpz', 'Cz', 'CPz'], ch_types='eeg')
    chinfo2 = ChInfo(ch_names=['Oz', 'Cz', 'CPz'], ch_types='eeg')
    with pytest.raises(ValueError, match='does not have the same channels'):
        _compare_infos(chinfo1, chinfo2)
    with pytest.raises(ValueError, match='does not have the same channels'):
        _compare_infos(chinfo2, chinfo1)

    # with subset of channels
    info1 = ChInfo(ch_names=['Fpz', 'Cz', 'CPz'], ch_types='eeg')
    info2 = ChInfo(ch_names=['Cz', 'CPz'], ch_types='eeg')
    with pytest.raises(ValueError, match='does not have the same channels'):
        _compare_infos(cluster_info=info1, inst_info=info2)
    _compare_infos(cluster_info=info2, inst_info=info1)

    # with different kind/unit/coord_frame
    chinfo1 = ChInfo(ch_names=['Fpz', 'Cz', 'CPz'], ch_types='eeg')
    chinfo1.set_montage('standard_1020')
    chinfo2 = chinfo1.copy()
    chinfo2['chs'][0]['unit'] = FIFF.FIFF_UNIT_C
    caplog.clear()
    _compare_infos(chinfo1, chinfo2)
    assert 'does not have the same channels units' in caplog.text
    caplog.clear()
    chinfo2 = chinfo1.copy()
    chinfo2['chs'][0]['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
    _compare_infos(chinfo1, chinfo2)
    assert 'does not have the same coordinate frames' in caplog.text
    caplog.clear()
    chinfo2 = chinfo1.copy()
    chinfo2['chs'][0]['kind'] = FIFF.FIFFV_MEG_CH
    _compare_infos(chinfo1, chinfo2)
    assert 'does not have the same channels kinds' in caplog.text
    caplog.clear()
    chinfo2 = chinfo1.copy()
    chinfo2['chs'][0]['unit'] = FIFF.FIFF_UNIT_C
    chinfo2['chs'][0]['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
    chinfo2['chs'][0]['kind'] = FIFF.FIFFV_MEG_CH
    _compare_infos(chinfo1, chinfo2)
    assert 'does not have the same channels units' in caplog.text
    assert 'does not have the same coordinate frames' in caplog.text
    assert 'does not have the same channels kinds' in caplog.text
