from mne.io.meas_info import ContainsMixin as MNE_ContainsMixin


class ContainsMixin(MNE_ContainsMixin):
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
