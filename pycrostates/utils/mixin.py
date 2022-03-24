"""Montage for pycrostates data container with a .info attribute."""

from mne.io.meas_info import ContainsMixin as MNEContainsMixin
from mne.io.meas_info import MontageMixin as MNEMontageMixin


class ChannelsMixin():
    # TODO: Maybe some part are salvageable from mne.channels.channels:
    # - UpdateChannelsMixin
    # - SetChannelsMixin
    pass


class ContainsMixin(MNEContainsMixin):
    def __getattribute__(self, name):
        """Attribute getter."""
        _req_info = (
            'compensation_grade',
            'get_channel_types',
            )

        if name in _req_info:
            if not hasattr(self, 'info'):
                raise ValueError(
                    f"Instance '{self.__class__.__name__}' is missing an "
                    f"attribute 'info' required by '{name}'."
                    )
            if self.info is None:
                raise ValueError(
                    f"Instance '{self.__class__.__name__}' attribute 'info' "
                    f"is None. An Info/ChInfo instance is required by {name}'."
                    )

        # invalid attributes/properties
        _inv_attributes = (
            )

        # invalid methods
        _inv_methods = (
            )

        # disable method/attributes that pycrostates does not support
        if name in _inv_attributes or name in _inv_methods:
            raise AttributeError(
                f"'{self.__class__.__name__}' has not attribute '{name}'")
        else:
            return super().__getattribute__(name)


class MontageMixin(MNEMontageMixin):
    def __getattribute__(self, name):
        """Attribute getter."""
        _req_info = (
            'get_montage',
            'set_montage',
            )

        if name in _req_info:
            if not hasattr(self, 'info'):
                raise ValueError(
                    f"Instance '{self.__class__.__name__}' is missing an "
                    f"attribute 'info' required by '{name}'."
                    )
            if self.info is None:
                raise ValueError(
                    f"Instance '{self.__class__.__name__}' attribute 'info' "
                    f"is None. An Info/ChInfo instance is required by {name}'."
                    )

        # invalid attributes/properties
        _inv_attributes = (
            )

        # invalid methods
        _inv_methods = (
            )

        # disable method/attributes that pycrostates does not support
        if name in _inv_attributes or name in _inv_methods:
            raise AttributeError(
                f"'{self.__class__.__name__}' has not attribute '{name}'")
        else:
            return super().__getattribute__(name)
