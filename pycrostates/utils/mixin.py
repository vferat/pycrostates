"""Mixins for pycrostates data container with a .info attribute."""

from mne.io.meas_info import ContainsMixin as MNEContainsMixin
from mne.io.meas_info import MontageMixin as MNEMontageMixin

from ._docs import copy_doc


class ChannelsMixin:
    """Channels Mixin for futur implementation."""

    # TODO: Maybe some part are salvageable from mne.channels.channels:
    # - UpdateChannelsMixin
    # - SetChannelsMixin
    pass


@copy_doc(MNEContainsMixin)
class ContainsMixin(MNEContainsMixin):
    @copy_doc(MNEContainsMixin.__contains__)
    def __contains__(self, ch_type):
        if not hasattr(self, "info"):
            raise ValueError(
                f"Instance '{self.__class__.__name__}' is missing an "
                "attribute 'info' required by 'in' operator."
            )
        if self.info is None:
            raise ValueError(
                f"Instance '{self.__class__.__name__}' attribute 'info' "
                "is None. An Info/ChInfo instance is required by 'in' "
                "operator."
            )
        return super().__contains__(ch_type)

    def __getattribute__(self, name):
        """Attribute getter."""
        # check if the attribute requires a .info to work
        _req_info = (
            "compensation_grade",
            "get_channel_types",
        )
        if name in _req_info:
            if not hasattr(self, "info"):
                raise ValueError(
                    f"Instance '{self.__class__.__name__}' is missing an "
                    f"attribute 'info' required by '{name}'."
                )
            if self.info is None:
                raise ValueError(
                    f"Instance '{self.__class__.__name__}' attribute 'info' "
                    "is None. An Info/ChInfo instance is required by "
                    f"'{name}'."
                )

        # disable method/attributes that pycrostates does not support
        # invalid attributes
        _inv_attributes = ()
        # invalid methods/properties
        _inv_methods = ()

        if name in _inv_attributes or name in _inv_methods:
            raise AttributeError(
                f"'{self.__class__.__name__}' has not attribute '{name}'"
            )
        return super().__getattribute__(name)


@copy_doc(MNEMontageMixin)
class MontageMixin(MNEMontageMixin):
    def __getattribute__(self, name):
        """Attribute getter."""
        # check if the attribute requires a .info to work
        _req_info = (
            "get_montage",
            "set_montage",
        )
        if name in _req_info:
            if not hasattr(self, "info"):
                raise ValueError(
                    f"Instance '{self.__class__.__name__}' is missing an "
                    f"attribute 'info' required by '{name}'."
                )
            if self.info is None:
                raise ValueError(
                    f"Instance '{self.__class__.__name__}' attribute 'info' "
                    "is None. An Info/ChInfo instance is required by "
                    f"'{name}'."
                )

        # disable method/attributes that pycrostates does not support
        # invalid attributes
        _inv_attributes = ()
        # invalid methods/properties
        _inv_methods = ()
        if name in _inv_attributes or name in _inv_methods:
            raise AttributeError(
                f"'{self.__class__.__name__}' has not attribute '{name}'"
            )
        else:
            return super().__getattribute__(name)
