"""Test meas_info.py"""

from collections import OrderedDict

import numpy as np
import pytest
from mne import Transform, create_info
from mne.channels import DigMontage
from mne.datasets import testing
from mne.io import read_raw_fif
from mne.io.constants import FIFF
from numpy.testing import assert_allclose

from pycrostates.io import ChInfo
from pycrostates.utils._logs import logger, set_log_level

set_log_level("INFO")
logger.propagate = True


directory = testing.data_path() / "MEG" / "sample"
fname = directory / "sample_audvis_trunc_raw.fif"
raw = read_raw_fif(fname, preload=False)


def test_create_from_info():
    """Test creation of a ChInfo from an Info instance."""
    info = create_info(ch_names=["1", "2", "3"], sfreq=1, ch_types="eeg")
    chinfo = ChInfo(info=info)

    # basic tests
    assert chinfo["bads"] == []
    assert chinfo["ch_names"] == ["1", "2", "3"]
    for k, ch in enumerate(chinfo["chs"]):
        assert ch["kind"] == FIFF.FIFFV_EEG_CH
        assert ch["coil_type"] == FIFF.FIFFV_COIL_EEG
        assert ch["unit"] == FIFF.FIFF_UNIT_V
        assert ch["coord_frame"] == FIFF.FIFFV_COORD_HEAD
        assert ch["ch_name"] == str(k + 1)
        assert all(np.isnan(elt) for elt in ch["loc"])
    assert chinfo["dig"] is None
    assert chinfo["custom_ref_applied"] == FIFF.FIFFV_MNE_CUSTOM_REF_OFF
    assert chinfo["nchan"] == 3
    assert chinfo["ctf_head_t"] is None
    assert chinfo["dev_ctf_t"] is None
    assert chinfo["dev_head_t"] == Transform("meg", "head")

    # test changing a value from info
    with info._unlock():
        info["custom_ref_applied"] = FIFF.FIFFV_MNE_CUSTOM_REF_ON
    assert chinfo["custom_ref_applied"] == FIFF.FIFFV_MNE_CUSTOM_REF_OFF

    # test adding bads
    assert len(chinfo["bads"]) == 0
    chinfo["bads"] = ["1"]
    assert len(chinfo["bads"]) == 1

    # test with info that has montage
    info = create_info(ch_names=["Fp1", "Fp2", "Fpz"], sfreq=1, ch_types="eeg")
    info.set_montage("standard_1020")
    chinfo = ChInfo(info=info)
    assert chinfo["ch_names"] == ["Fp1", "Fp2", "Fpz"]
    for ch in info["chs"]:
        assert ch["coord_frame"] == FIFF.FIFFV_COORD_HEAD
        assert not all(np.isnan(elt) for elt in ch["loc"])
    assert chinfo["dig"] is not None
    assert chinfo["nchan"] == 3
    assert chinfo["ctf_head_t"] is None
    assert chinfo["dev_ctf_t"] is None
    assert chinfo["dev_head_t"] == Transform("meg", "head")

    # test with multiple channel types
    ch_names = [f"MEG{n:03}" for n in range(1, 10)] + ["EOG001"]
    ch_types = ["mag", "grad", "grad"] * 3 + ["eog"]
    kinds_ = {
        "mag": FIFF.FIFFV_MEG_CH,
        "grad": FIFF.FIFFV_MEG_CH,
        "eog": FIFF.FIFFV_EOG_CH,
    }
    coil_type_ = {
        "mag": FIFF.FIFFV_COIL_VV_MAG_T3,
        "grad": FIFF.FIFFV_COIL_VV_PLANAR_T1,
        "eog": FIFF.FIFFV_COIL_NONE,
    }
    unit_ = {
        "mag": FIFF.FIFF_UNIT_T,
        "grad": FIFF.FIFF_UNIT_T_M,
        "eog": FIFF.FIFF_UNIT_V,
    }
    info = create_info(ch_names, ch_types=ch_types, sfreq=1)
    chinfo = ChInfo(info=info)
    assert chinfo["ch_names"] == ch_names
    for k, ch in enumerate(chinfo["chs"]):
        assert ch["kind"] == kinds_[ch_types[k]]
        assert ch["coil_type"] == coil_type_[ch_types[k]]
        assert ch["unit"] == unit_[ch_types[k]]
        if "EOG" in ch["ch_name"]:
            assert ch["coord_frame"] == FIFF.FIFFV_COORD_UNKNOWN
        else:
            assert ch["coord_frame"] == FIFF.FIFFV_COORD_DEVICE
        assert ch["ch_name"] == ch_names[k]
        assert all(np.isnan(elt) for elt in ch["loc"])

    # test with a file with projs and coordinate transformation
    chinfo = ChInfo(info=raw.info)
    assert chinfo["ctf_head_t"] is None
    assert chinfo["dev_ctf_t"] is None
    assert chinfo["dev_head_t"] == raw.info["dev_head_t"]


def test_create_from_info_invalid_arguments():
    """Test creation of a ChInfo from an Info instance with invalid args."""
    ch_names = ["1", "2", "3"]
    ch_types = ["eeg"] * 3
    info = create_info(ch_names=ch_names, sfreq=1, ch_types=ch_types)

    with pytest.raises(RuntimeError, match="If 'info' is provided"):
        ChInfo(info, ch_names=ch_names)
    with pytest.raises(RuntimeError, match="If 'info' is provided"):
        ChInfo(info, ch_types=ch_types)
    with pytest.raises(TypeError, match="'info' must be an instance of None or Info"):
        ChInfo(info=ch_names)


def test_create_from_channels():
    """Test creation of a ChInfo from channel names and types."""
    ch_names = ["1", "2", "3"]
    ch_types = ["eeg"] * 3
    chinfo = ChInfo(ch_names=ch_names, ch_types=ch_types)

    # basic tests
    assert chinfo["bads"] == []
    assert chinfo["ch_names"] == ["1", "2", "3"]
    for k, ch in enumerate(chinfo["chs"]):
        assert ch["kind"] == FIFF.FIFFV_EEG_CH
        assert ch["coil_type"] == FIFF.FIFFV_COIL_EEG
        assert ch["unit"] == FIFF.FIFF_UNIT_V
        assert ch["coord_frame"] == FIFF.FIFFV_COORD_HEAD
        assert ch["ch_name"] == str(k + 1)
        assert all(np.isnan(elt) for elt in ch["loc"])
    assert chinfo["dig"] is None
    assert chinfo["custom_ref_applied"] == FIFF.FIFFV_MNE_CUSTOM_REF_OFF
    assert chinfo["nchan"] == 3
    assert chinfo["projs"] == []

    # test with multiple channel types
    ch_names = [f"MEG{n:03}" for n in range(1, 10)] + ["EOG001"]
    ch_types = ["mag", "grad", "grad"] * 3 + ["eog"]
    kinds_ = {
        "mag": FIFF.FIFFV_MEG_CH,
        "grad": FIFF.FIFFV_MEG_CH,
        "eog": FIFF.FIFFV_EOG_CH,
    }
    coil_type_ = {
        "mag": FIFF.FIFFV_COIL_VV_MAG_T3,
        "grad": FIFF.FIFFV_COIL_VV_PLANAR_T1,
        "eog": FIFF.FIFFV_COIL_NONE,
    }
    unit_ = {
        "mag": FIFF.FIFF_UNIT_T,
        "grad": FIFF.FIFF_UNIT_T_M,
        "eog": FIFF.FIFF_UNIT_V,
    }
    chinfo = ChInfo(ch_names=ch_names, ch_types=ch_types)
    assert chinfo["ch_names"] == ch_names
    for k, ch in enumerate(chinfo["chs"]):
        assert ch["kind"] == kinds_[ch_types[k]]
        assert ch["coil_type"] == coil_type_[ch_types[k]]
        assert ch["unit"] == unit_[ch_types[k]]
        if "EOG" in ch["ch_name"]:
            assert ch["coord_frame"] == FIFF.FIFFV_COORD_UNKNOWN
        else:
            assert ch["coord_frame"] == FIFF.FIFFV_COORD_DEVICE
        assert ch["ch_name"] == ch_names[k]
        assert all(np.isnan(elt) for elt in ch["loc"])

    # test with ch_names as int
    ch_names = 3
    ch_types = ["eeg"] * 3
    chinfo = ChInfo(ch_names=ch_names, ch_types=ch_types)

    # basic tests
    assert chinfo["bads"] == []
    assert chinfo["ch_names"] == ["0", "1", "2"]
    for k, ch in enumerate(chinfo["chs"]):
        assert ch["kind"] == FIFF.FIFFV_EEG_CH
        assert ch["coil_type"] == FIFF.FIFFV_COIL_EEG
        assert ch["unit"] == FIFF.FIFF_UNIT_V
        assert ch["coord_frame"] == FIFF.FIFFV_COORD_HEAD
        assert ch["ch_name"] == str(k)
        assert all(np.isnan(elt) for elt in ch["loc"])
    assert chinfo["dig"] is None
    assert chinfo["custom_ref_applied"] == FIFF.FIFFV_MNE_CUSTOM_REF_OFF
    assert chinfo["nchan"] == 3


def test_create_from_channels_invalid_arguments():
    """Test creation of a ChInfo from channel names and types with invalid
    args."""
    ch_names = ["1", "2", "3"]
    ch_types = ["eeg"] * 3
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
    with pytest.raises(
        RuntimeError,
        match="Either 'info' or 'ch_names' and 'ch_types' must not be None.",
    ):
        ChInfo()


def test_montage():
    """Test methods from montage Mixin."""
    info = create_info(ch_names=["Fp1", "Fp2", "Fpz"], sfreq=1, ch_types="eeg")
    info.set_montage("standard_1020")
    chinfo = ChInfo(info=info)
    assert chinfo["ch_names"] == ["Fp1", "Fp2", "Fpz"]
    for ch in info["chs"]:
        assert ch["coord_frame"] == FIFF.FIFFV_COORD_HEAD
        assert not all(np.isnan(elt) for elt in ch["loc"])
    assert chinfo["dig"] is not None
    assert chinfo["nchan"] == 3

    # retrieve montage
    montage = chinfo.get_montage()
    assert isinstance(montage, DigMontage)
    assert montage.ch_names == ["Fp1", "Fp2", "Fpz"]

    # add retrieved montage to different ChInfo
    chinfo2 = ChInfo(ch_names=["Fp1", "Fp2", "Fpz"], ch_types="eeg")
    assert chinfo2["dig"] is None
    montage2 = chinfo2.get_montage()
    assert montage2 is None
    chinfo2.set_montage(montage)
    montage2 = chinfo2.get_montage()
    assert isinstance(montage2, DigMontage)
    assert montage2.ch_names == ["Fp1", "Fp2", "Fpz"]

    # compare positions
    assert sorted(list(montage.get_positions().keys())) == sorted(
        list(montage2.get_positions().keys())
    )
    for key in sorted(list(montage.get_positions().keys())):
        if montage.get_positions()[key] is None:
            assert montage2.get_positions()[key] is None
        elif isinstance(montage.get_positions()[key], str):
            assert montage.get_positions()[key] == montage2.get_positions()[key]
        elif isinstance(montage.get_positions()[key], np.ndarray):
            assert_allclose(montage.get_positions()[key], montage2.get_positions()[key])
        elif isinstance(montage.get_positions()[key], OrderedDict):
            for k in montage.get_positions()[key]:
                assert_allclose(
                    montage.get_positions()[key][k],
                    montage2.get_positions()[key][k],
                )


def test_contains():
    """Test methods from contain Mixin."""
    info = create_info(ch_names=["Fp1", "Fp2", "Fpz"], sfreq=1, ch_types="eeg")
    info.set_montage("standard_1020")
    chinfo = ChInfo(info=info)
    assert chinfo.get_channel_types() == ["eeg"] * 3
    assert chinfo.compensation_grade is None


def test_copy():
    """Test copy (which is a deepcopy)."""
    info = create_info(ch_names=["Fp1", "Fp2", "Fpz"], sfreq=1, ch_types="eeg")
    info.set_montage("standard_1020")
    chinfo = ChInfo(info=info)
    chinfo2 = chinfo.copy()
    with chinfo._unlock():
        chinfo["custom_ref_applied"] = FIFF.FIFFV_MNE_CUSTOM_REF_ON
    assert chinfo["custom_ref_applied"] == FIFF.FIFFV_MNE_CUSTOM_REF_ON
    assert chinfo2["custom_ref_applied"] == FIFF.FIFFV_MNE_CUSTOM_REF_OFF


def test_repr():
    """Test normal and HTML representation."""
    info = create_info(ch_names=["Fp1", "Fp2", "Fpz"], sfreq=1, ch_types="eeg")
    info.set_montage("standard_1020")
    chinfo = ChInfo(info=info)

    # normal repr
    repr_ = chinfo.__repr__()
    assert "Info | 4 non-empty values" in repr_
    assert "bads: []" in repr_
    assert "ch_names: Fp1, Fp2, Fpz" in repr_
    assert "chs: 3 EEG" in repr_
    assert "custom_ref_applied: False" in repr_
    assert "dig: 6 items (3 Cardinal, 3 EEG)" in repr_
    assert "nchan: 3" in repr_

    # html repr
    chinfo._repr_html_()
    # TODO: Needs more test and probably an overwrite of _repr_html_().


def test_setting_invalid_keys():
    """Test raise when invalid keys are set. Test locking mechanism."""
    info = create_info(ch_names=3, sfreq=1, ch_types="eeg")
    chinfo = ChInfo(info=info)

    with pytest.raises(
        RuntimeError, match="Info does not support directly setting the key 'test'"
    ):
        chinfo["test"] = 5

    with pytest.raises(RuntimeError, match="ch_names cannot be set directly."):
        chinfo["ch_names"] = ["4", "5", "6"]

    with chinfo._unlock():
        chinfo["ch_names"] = ["4", "5", "6"]
    assert chinfo["ch_names"] == ["4", "5", "6"]

    with pytest.raises(RuntimeError, match="info channel name inconsistency"):
        chinfo._check_consistency()


def test_comparison(caplog):
    """Test == and != methods."""
    # simple info without montage
    info1 = create_info(ch_names=3, sfreq=1, ch_types="eeg")
    info2 = create_info(ch_names=3, sfreq=1, ch_types="eeg")
    chinfo1 = ChInfo(info1)
    chinfo2 = ChInfo(info2)
    assert chinfo1 == chinfo2
    chinfo1["bads"] = [chinfo1["ch_names"][0]]
    caplog.clear()
    assert chinfo1 == chinfo2
    assert "Both info do not have the same bad channels." in caplog.text
    info3 = create_info(ch_names=4, sfreq=1, ch_types="eeg")
    chinfo3 = ChInfo(info3)
    assert chinfo2 != chinfo3

    # with montage
    chinfo1 = ChInfo(ch_names=["Cz", "Oz"], ch_types="eeg")
    chinfo1.set_montage("standard_1020")
    chinfo2 = chinfo1.copy()
    assert chinfo1 == chinfo2
    chinfo2 = ChInfo(ch_names=["Cz", "Oz"], ch_types="eeg")
    assert chinfo1 != chinfo2

    # with different channel types
    chinfo1 = ChInfo(ch_names=["Cz", "Oz"], ch_types="eeg")
    chinfo2 = ChInfo(ch_names=["Cz", "Oz"], ch_types="misc")
    assert chinfo1 != chinfo2

    # with different ref
    chinfo1 = ChInfo(raw.info)
    chinfo2 = ChInfo(raw.info)
    with chinfo2._unlock():
        chinfo2["custom_ref_applied"] = FIFF.FIFFV_MNE_CUSTOM_REF_ON
    assert chinfo1 != chinfo2

    # with different object
    assert chinfo1 != 101

    # with projs
    chinfo1 = ChInfo(raw.info)
    chinfo2 = ChInfo(raw.info)
    assert chinfo1 == chinfo2
    with chinfo1._unlock():
        chinfo1["projs"] = []
    assert chinfo1 != chinfo2
