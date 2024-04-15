"""Class to handle no temporal but spatial data."""

from __future__ import annotations  # c.f. PEP 563, PEP 649

from copy import copy, deepcopy
from typing import TYPE_CHECKING

import numpy as np
from mne import Info, pick_info
from mne.utils import check_version

if check_version("mne", "1.6"):
    from mne._fiff.pick import _picks_to_idx
else:
    from mne.io.pick import _picks_to_idx

from ..utils._checks import _check_type
from ..utils._docs import fill_doc
from ..utils.mixin import ChannelsMixin, ContainsMixin, MontageMixin

if TYPE_CHECKING:
    from typing import Any, Union

    from .._typing import ScalarFloatArray
    from . import ChInfo


class ChData(ChannelsMixin, ContainsMixin, MontageMixin):
    """ChData stores atemporal data with its spatial information.

    `~pycrostates.io.ChData` is similar to a raw instance where temporality has been
    removed. Only the spatial information, stored as a `~pycrostates.io.ChInfo` is
    retained.

    Parameters
    ----------
    data : array
        Data array of shape ``(n_channels, n_samples)``.
    info : mne.Info | ChInfo
        Atemporal measurement information. If a `mne.Info` is provided, it is converted
        to a `~pycrostates.io.ChInfo`.
    """

    def __init__(self, data: ScalarFloatArray, info: Union[Info, ChInfo]):
        from . import ChInfo

        _check_type(data, (np.ndarray,), "data")
        _check_type(info, (Info, ChInfo), "info")
        if data.ndim != 2:
            raise ValueError(
                "Argument 'data' should be a 2D array (n_channels, n_samples). The "
                f"provided array shape is {data.shape} which has {data.ndim} "
                "dimensions."
            )
        if not len(info["ch_names"]) == data.shape[0]:
            raise ValueError(
                "Argument 'data' and 'info' do not have the same number of channels."
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
        info_repr = self._info._repr_html_()  # pylint: disable=protected-access
        return template.render(n_samples=self._data.shape[-1], info_repr=info_repr)

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
    def get_data(self, picks=None) -> ScalarFloatArray:
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

    @fill_doc
    def pick(self, picks, exclude="bads"):
        """Pick a subset of channels.

        Parameters
        ----------
        %(picks_all)s
        exclude : list | str
            Set of channels to exclude, only used when picking based on types (e.g.,
            ``exclude="bads"`` when ``picks="meg"``).

        Returns
        -------
        inst : ChData
            The instance modified in-place.
        """
        picks = _picks_to_idx(self._info, picks, exclude=exclude)
        info = pick_info(self._info, picks, copy=False)
        data = self._data[picks, :]
        self._info = info
        self._data = data
        return self

    def _get_channel_positions(self, picks=None):
        """Get channel locations from info.

        Parameters
        ----------
        picks : str | list | slice | None
            None selects the good data channels.

        Returns
        -------
        pos : array of shape (n_channels, 3)
            Channel X/Y/Z locations.
        """
        picks = _picks_to_idx(self.info, picks)
        chs = self.info["chs"]
        pos = np.array([chs[k]["loc"][:3] for k in picks])
        n_zero = np.sum(np.sum(np.abs(pos), axis=1) == 0)
        if n_zero > 1:  # XXX some systems have origin (0, 0, 0)
            raise ValueError(
                f"Could not extract channel positions for {n_zero} channels."
            )
        return pos

    # --------------------------------------------------------------------
    @property
    def info(self) -> ChInfo:
        """Atemporal measurement information.

        :type: ChInfo
        """
        return self._info

    @property
    def ch_names(self):
        """Channel names."""
        return self.info["ch_names"]

    @property
    def preload(self):
        """Preload required by some MNE functions."""
        return True
