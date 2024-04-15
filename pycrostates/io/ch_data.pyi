from typing import Any

from _typeshed import Incomplete
from mne import Info

from .._typing import ScalarFloatArray as ScalarFloatArray
from ..utils._checks import _check_type as _check_type
from ..utils._docs import fill_doc as fill_doc
from ..utils.mixin import ChannelsMixin as ChannelsMixin
from ..utils.mixin import ContainsMixin as ContainsMixin
from ..utils.mixin import MontageMixin as MontageMixin
from . import ChInfo as ChInfo

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

    _data: Incomplete
    _info: Incomplete

    def __init__(self, data: ScalarFloatArray, info: Info | ChInfo) -> None: ...
    def __repr__(self) -> str:
        """String representation."""

    def _repr_html_(self, caption: Incomplete | None = None):
        """HTML representation."""

    def __eq__(self, other: Any) -> bool:
        """Equality == method."""

    def __ne__(self, other: Any) -> bool:
        """Different != method."""

    def copy(self, deep: bool = True):
        """Return a copy of the instance.

        Parameters
        ----------
        deep : bool
            If True, `~copy.deepcopy` is used instead of `~copy.copy`.
        """

    def get_data(self, picks: Incomplete | None = None) -> ScalarFloatArray:
        """Retrieve the data array.

        Parameters
        ----------
        picks : str | array-like | slice | None
            Channels to include. Slices and lists of integers will be interpreted as
            channel indices. In lists, channel *type* strings (e.g., ``['meg',
            'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
            ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
            string values "all" to pick all channels, or "data" to pick :term:`data
            channels`. None (default) will pick all channels. Note that channels in
            ``info['bads']`` *will be included* if their names or indices are
            explicitly provided.

        Returns
        -------
        data : array
            Data array of shape ``(n_channels, n_samples)``.
        """

    def pick(self, picks, exclude: str = "bads"):
        """Pick a subset of channels.

        Parameters
        ----------
        picks : str | array-like | slice | None
            Channels to include. Slices and lists of integers will be interpreted as
            channel indices. In lists, channel *type* strings (e.g., ``['meg',
            'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
            ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
            string values "all" to pick all channels, or "data" to pick :term:`data
            channels`. None (default) will pick all channels. Note that channels in
            ``info['bads']`` *will be included* if their names or indices are
            explicitly provided.
        exclude : list | str
            Set of channels to exclude, only used when picking based on types (e.g.,
            ``exclude="bads"`` when ``picks="meg"``).

        Returns
        -------
        inst : ChData
            The instance modified in-place.
        """

    def _get_channel_positions(self, picks: Incomplete | None = None):
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

    @property
    def info(self) -> ChInfo:
        """Atemporal measurement information.

        :type: ChInfo
        """

    @property
    def ch_names(self):
        """Channel names."""

    @property
    def preload(self):
        """Preload required by some MNE functions."""
