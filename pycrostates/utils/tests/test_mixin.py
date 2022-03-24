"""Test mixins.py"""

from pycrostates.io import ChInfo
from pycrostates.utils.mixin import ContainsMixin


class Foo(ContainsMixin):
    def __init__(self, info):
        self.info = info


def test_contains_mixin():
    """Test ContainsMixin."""
    ch_types = ['eeg', 'eeg', 'grad', 'grad', 'mag']
    info = ChInfo(ch_names=5, ch_types=ch_types)
    foo = Foo(info)
    assert 'eeg' in foo
    assert 'meg' in foo
    assert 'grad' in foo
    assert 'mag' in foo
    assert foo.compensation_grade == 0
    assert foo.get_channel_types() == ch_types

    ch_types = ['eeg'] * 5
    info = ChInfo(ch_names=5, ch_types=ch_types)
    foo = Foo(info)
    assert 'eeg' in foo
    assert foo.compensation_grade is None
    assert foo.get_channel_types() == ch_types
