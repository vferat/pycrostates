"""Clas to handle no temporal but spatial data."""
from copy import copy, deepcopy
from typing import Any, Union

import numpy as np
from mne.io import Info
from mne.io.pick import _picks_to_idx
from numpy.typing import NDArray

from .._typing import CHData, CHInfo
from ..utils._checks import _check_type
from ..utils._docs import fill_doc
from ..utils.mixin import ChannelsMixin, ContainsMixin, MontageMixin


class ChData(CHData, ChannelsMixin, ContainsMixin, MontageMixin):
    """ChData stores atemporal data with its spatial information.

    ChData is similar to a raw instance where temporality has been removed.
    Only the spatial information, stored as a `~pycrostates.io.ChInfo` is
    retained.

    Parameters
    ----------
    data : array
        Data array of shape ``(n_channels, n_samples)``.
    info : mne.Info | ChInfo
        Atemporal measurement information. If a `mne.Info` is provided, it is
        converted to a `~pycrostates.io.ChInfo`.
    """

    def __init__(self, data: NDArray[float], info: Union[Info, CHInfo]):
        from .meas_info import ChInfo

        _check_type(data, (np.ndarray,), "data")
        _check_type(info, (Info, ChInfo), "info")
        if data.ndim != 2:
            raise ValueError(
                "Argument 'data' should be a 2D array "
                "(n_channels, n_samples). The provided array "
                f"shape is {data.shape} which has {data.ndim} "
                "dimensions."
            )
        if not len(info["ch_names"]) == data.shape[0]:
            raise ValueError(
                "Argument 'data' and 'info' do not have the same "
                "number of channels."
            )
        self._data = data
        self._info = info if isinstance(info, ChInfo) else ChInfo(info)

    def __repr__(self) -> str:  # noqa: D401
        """String representation."""
        name = self.__class__.__name__
        s = f"< {name} | {self._data.shape[-1]} samples >"
        return s

    def _repr_html_(self, caption=None):
        """HTML representation."""
        from ..html_templates import repr_templates_env

        template = repr_templates_env.get_template("ChData.html.jinja")
        info_repr = (
            self._info._repr_html_()
        )  # pylint: disable=protected-access
        return template.render(
            n_samples=self._data.shape[-1], info_repr=info_repr
        )

    def __eq__(self, other: Any) -> bool:
        """Equality == method."""
        if (
            isinstance(other, ChData)
            and other.info == self.info
            and other._data.shape == self._data.shape
            and np.allclose(other._data, self._data)
        ):
            return True
        else:
            return False

    def __ne__(self, other: Any) -> bool:
        """Different != method."""
        return not self.__eq__(other)

    def copy(
        self,
        deep: bool = True,
    ):
        """Return a copy of the instance.

        Parameters
        ----------
        deep : bool
            If True, `~copy.deepcopy` is used instead of `~copy.copy`.
        """
        if deep:
            return deepcopy(self)
        return copy(self)

    @fill_doc
    def get_data(self, picks=None) -> NDArray[float]:
        """Retrieve the data array.

        Parameters
        ----------
        %(picks_all)s

        Returns
        -------
        data : array
            Data array of shape ``(n_channels, n_samples)``.
        """
        picks = _picks_to_idx(self._info, picks, none="all", exclude="bads")
        data = self._data.copy()
        return data[picks, :]

    # --------------------------------------------------------------------
    @property
    def info(self) -> CHInfo:
        """Atemporal measurement information.

        :type: `~pycrostates.io.ChInfo`
        """
        return self._info
