"""Test mixins.py"""

import pytest
from mne.channels import DigMontage

from pycrostates.io import ChInfo
from pycrostates.utils.mixin import ContainsMixin, MontageMixin


class Foo(ContainsMixin, MontageMixin):
    def __init__(self, info):
        self.info = info


class Foo2(ContainsMixin, MontageMixin):
    def __init__(self):
        pass


def test_contains_mixin():
    """Test ContainsMixin."""
    ch_types = ["eeg", "eeg", "grad", "grad", "mag"]
    info = ChInfo(ch_names=5, ch_types=ch_types)
    foo = Foo(info)
    assert "eeg" in foo
    assert "meg" in foo
    assert "grad" in foo
    assert "mag" in foo
    assert foo.compensation_grade == 0
    assert foo.get_channel_types() == ch_types

    ch_types = ["eeg"] * 5
    info = ChInfo(ch_names=5, ch_types=ch_types)
    foo = Foo(info)
    assert "eeg" in foo
    assert foo.compensation_grade is None
    assert foo.get_channel_types() == ch_types

    # test with info equal to None
    foo = Foo(None)
    with pytest.raises(ValueError, match="Instance 'Foo' attribute 'info' is None."):
        "eeg" in foo  # noqa: B015
    with pytest.raises(ValueError, match="Instance 'Foo' attribute 'info' is None."):
        foo.get_channel_types()
    with pytest.raises(ValueError, match="Instance 'Foo' attribute 'info' is None."):
        foo.compensation_grade  # noqa: B018

    # test without attribute info
    foo = Foo2()
    with pytest.raises(
        ValueError, match="Instance 'Foo2' is missing an attribute 'info'"
    ):
        "eeg" in foo  # noqa: B015
    with pytest.raises(
        ValueError, match="Instance 'Foo2' is missing an attribute 'info'"
    ):
        foo.get_channel_types()
    with pytest.raises(
        ValueError, match="Instance 'Foo2' is missing an attribute 'info'"
    ):
        foo.compensation_grade  # noqa: B018


def test_montage_mixin():
    """Test MontageMixin."""
    info = ChInfo(ch_names=["Fp1", "Fpz", "Fp2"], ch_types="eeg")
    assert info.get_montage() is None

    foo = Foo(info)
    assert foo.info["dig"] is None
    foo.set_montage("standard_1020")
    assert foo.info["dig"] is not None

    montage = foo.get_montage()
    assert isinstance(montage, DigMontage)

    # test with info equal to None
    foo = Foo(None)
    with pytest.raises(ValueError, match="Instance 'Foo' attribute 'info' is None."):
        foo.set_montage("standard_1020")
    with pytest.raises(ValueError, match="Instance 'Foo' attribute 'info' is None."):
        foo.get_montage()

    # test without attribute info
    foo = Foo2()
    with pytest.raises(
        ValueError, match="Instance 'Foo2' is missing an attribute 'info'"
    ):
        foo.set_montage("standard_1020")
    with pytest.raises(
        ValueError, match="Instance 'Foo2' is missing an attribute 'info'"
    ):
        foo.get_montage()
