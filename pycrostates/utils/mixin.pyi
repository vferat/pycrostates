from mne.io.meas_info import ContainsMixin as MNEContainsMixin
from mne.io.meas_info import MontageMixin as MNEMontageMixin

from ._docs import copy_doc as copy_doc

class ChannelsMixin:
    """Channels Mixin for futur implementation."""

class ContainsMixin(MNEContainsMixin):
    def __contains__(self, ch_type) -> bool: ...
    def __getattribute__(self, name):
        """Attribute getter."""

class MontageMixin(MNEMontageMixin):
    def __getattribute__(self, name):
        """Attribute getter."""
