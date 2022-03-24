from mne.channels.channels import UpdateChannelsMixin as MNEUpdateChannelsMixin
from mne.channels.channels import SetChannelsMixin as MNESetChannelsMixin
from mne.io.meas_info import ContainsMixin as MNEContainsMixin


class UpdateChannelsMixin(MNEUpdateChannelsMixin):
    def __getattribute__(self, name):
        """Attribute getter."""
        # invalid attributes/properties
        _attributes = (
            )

        # invalid methods
        _methods = (
            )

        # disable method/attributes that we do not support
        if name in _attributes or name in _methods:
            raise AttributeError(
                f"'{self.__class__.__name__}' has not attribute '{name}'")
        else:
            return super().__getattribute__(name)


class SetChannelsMixin(MNESetChannelsMixin):
    def __getattribute__(self, name):
        """Attribute getter."""
        # invalid attributes/properties
        _attributes = (
            )

        # invalid methods
        _methods = (
            )

        # disable method/attributes that we do not support
        if name in _attributes or name in _methods:
            raise AttributeError(
                f"'{self.__class__.__name__}' has not attribute '{name}'")
        else:
            return super().__getattribute__(name)


class ContainsMixin(MNEContainsMixin):
    def __getattribute__(self, name):
        """Attribute getter."""
        # invalid attributes/properties
        _attributes = (
            )

        # invalid methods
        _methods = (
            )

        # disable method/attributes that we do not support
        if name in _attributes or name in _methods:
            raise AttributeError(
                f"'{self.__class__.__name__}' has not attribute '{name}'")
        else:
            return super().__getattribute__(name)
