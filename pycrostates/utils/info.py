from copy import deepcopy
from operator import index

from mne.transforms import _frame_to_str
from mne.io import Info
from mne.io.compensator import get_current_comp
from mne.io.constants import FIFF
from mne.io.pick import (get_channel_type_constants, pick_types,
                         _contains_ch_type, _get_channel_types)
from mne.io.tag import _ch_coord_dict
from mne.io._digitization import _get_data_as_dict_from_dig
import numpy as np

from ._checks import _check_type, _IntLike
from ._docs import fill_doc
from ._logs import verbose


# TODO: Copying the Mixin is a pain.. It should be possible to fool isinstance
# in believing that `ChInfo` is an `Info` instance by giving to `ChInfo` only
# the 'type' from `Info`.
# https://stackoverflow.com/questions/71505152/inherit-only-the-type-from-parent-class-in-python
class MontageMixin:
    """Mixin for Montage getting and setting."""

    @fill_doc
    def get_montage(self):
        """Get a DigMontage from instance.

        Returns
        -------
        %(montage)s
        """
        from mne.channels.montage import make_dig_montage
        info = self if isinstance(self, ChInfo) else self.info
        if info['dig'] is None:
            return None
        # obtain coord_frame, and landmark coords
        # (nasion, lpa, rpa, hsp, hpi) from DigPoints
        montage_bunch = _get_data_as_dict_from_dig(info['dig'])
        coord_frame = _frame_to_str.get(montage_bunch.coord_frame)

        # get the channel names and chs data structure
        ch_names, chs = info['ch_names'], info['chs']
        picks = pick_types(info, meg=False, eeg=True, seeg=True,
                           ecog=True, dbs=True, fnirs=True, exclude=[])

        # channel positions from dig do not match ch_names one to one,
        # so use loc[:3] instead
        ch_pos = {ch_names[ii]: chs[ii]['loc'][:3] for ii in picks}

        # create montage
        montage = make_dig_montage(
            ch_pos=ch_pos,
            coord_frame=coord_frame,
            nasion=montage_bunch.nasion,
            lpa=montage_bunch.lpa,
            rpa=montage_bunch.rpa,
            hsp=montage_bunch.hsp,
            hpi=montage_bunch.hpi,
        )
        return montage

    @verbose
    def set_montage(self, montage, match_case=True, match_alias=False,
                    on_missing='raise', verbose=None):
        """Set %(montage_types)s channel positions and digitization points.

        Parameters
        ----------
        %(montage)s
        %(match_case)s
        %(match_alias)s
        %(on_missing_montage)s
        %(verbose)s

        Returns
        -------
        inst : instance of ChInfo | Cluster | ChData
            The instance.

        Notes
        -----
        Operates in place.

        .. warning::
            Only %(montage_types)s channels can have their positions set using
            a montage. Other channel types (e.g., MEG channels) should have
            their positions defined properly using their data reading
            functions.
        """
        from mne.channels.montage import _set_montage
        info = self if isinstance(self, Info) else self.info
        _set_montage(info, montage, match_case, match_alias, on_missing)
        return self


class ContainsMixin:
    """Mixin class for channel types."""

    def __contains__(self, ch_type):
        """Check channel type membership.

        Parameters
        ----------
        ch_type : str
            Channel type to check for. Can be e.g. 'meg', 'eeg', 'stim', etc.

        Returns
        -------
        in : bool
            Whether or not the instance contains the given channel type.

        Examples
        --------
        Channel type membership can be tested as::

            >>> 'meg' in inst  # doctest: +SKIP
            True
            >>> 'seeg' in inst  # doctest: +SKIP
            False

        """
        info = self if isinstance(self, ChInfo) else self.info
        if ch_type == 'meg':
            has_ch_type = (_contains_ch_type(info, 'mag') or
                           _contains_ch_type(info, 'grad'))
        else:
            has_ch_type = _contains_ch_type(info, ch_type)
        return has_ch_type

    @property
    def compensation_grade(self):
        """The current gradient compensation grade."""
        info = self if isinstance(self, ChInfo) else self.info
        return get_current_comp(info)

    @fill_doc
    def get_channel_types(self, picks=None, unique=False, only_data_chs=False):
        """Get a list of channel type for each channel.

        Parameters
        ----------
        %(picks_all)s
        unique : bool
            Whether to return only unique channel types. Default is ``False``.
        only_data_chs : bool
            Whether to ignore non-data channels. Default is ``False``.

        Returns
        -------
        channel_types : list
            The channel types.
        """
        info = self if isinstance(self, ChInfo) else self.info
        return _get_channel_types(info, picks=picks, unique=unique,
                                  only_data_chs=only_data_chs)


# TODO: Add the locking/unlocking mechanism to prevent users from directly
# modifying entries in ChInfo.
# TODO: Add == and != method to compare ChInfo with other ChInfo and with other
# mne Info.
# TODO: Add __repr__ method to summarize info.
class ChInfo(dict, MontageMixin, ContainsMixin):
    """Measurement information.

    Similar to a mne.Info class, but without any temporal information. Only the
    channel-related information are present. A ChInfo can be created either:
        - by providing an mne.Info class from which information are retrieved.
        - by providing the ch_names and the ch_types to create a new instance
    Only one of those 2 methods should be used at once.

    .. warning:: The only entriy that should be manually changed by the user
                 is ``info['bads']``. All other entries should be
                 considered read-only, though they can be modified by various
                 functions or methods (which have safeguards to ensure all
                 fields remain in sync).

    Parameters
    ----------
    info : mne.Info | None
        A mne measurement information instance from which channel-related
        variables are retrieved.
    ch_names : list of str | int | None
        Channel names. If an int, a list of channel names will be created
        from ``range(ch_names)``.
    ch_types : list of str | str | None
        Channel types. If str, then all channels are assumed to be of the same
        type.

    Attributes
    ----------
    bads : list of str
        List of bad (noisy/broken) channels, by name. These channels will by
        default be ignored by many processing steps.
    ch_names : tuple of str
        The names of the channels.
    chs : tuple of dict
        A list of channel information dictionaries, one per channel.
        See Notes for more information.
    custom_ref_applied : int
        Whether a custom (=other than average) reference has been applied to
        the EEG data. This flag is checked by some algorithms that require an
        average reference to be set.
    dig : tuple of dict | None
        The Polhemus digitization data in head coordinates.
        See Notes for more information.
    nchan : int
        Number of channels.

    Notes
    -----
    The following parameters have a nested structure.

    * ``chs`` list of dict:

        cal : float
            The calibration factor to bring the channels to physical
            units. Used in product with ``range`` to scale the data read
            from disk.
        ch_name : str
            The channel name.
        coil_type : int
            Coil type, e.g. ``FIFFV_COIL_MEG``.
        coord_frame : int
            The coordinate frame used, e.g. ``FIFFV_COORD_HEAD``.
        kind : int
            The kind of channel, e.g. ``FIFFV_EEG_CH``.
        loc : array, shape (12,)
            Channel location. For MEG this is the position plus the
            normal given by a 3x3 rotation matrix. For EEG this is the
            position followed by reference position (with 6 unused).
            The values are specified in device coordinates for MEG and in
            head coordinates for EEG channels, respectively.
        logno : int
            Logical channel number, conventions in the usage of this
            number vary.
        range : float
            The hardware-oriented part of the calibration factor.
            This should be only applied to the continuous raw data.
            Used in product with ``cal`` to scale data read from disk.
        scanno : int
            Scanning order number, starting from 1.
        unit : int
            The unit to use, e.g. ``FIFF_UNIT_T_M``.
        unit_mul : int
            Unit multipliers, most commonly ``FIFF_UNITM_NONE``.

     * ``dig`` list of dict:

         kind : int
             The kind of channel,
             e.g. ``FIFFV_POINT_EEG``, ``FIFFV_POINT_CARDINAL``.
         r : array, shape (3,)
             3D position in m. and coord_frame.
         ident : int
             Number specifying the identity of the point.
             e.g. ``FIFFV_POINT_NASION`` if kind is ``FIFFV_POINT_CARDINAL``, or
             42 if kind is ``FIFFV_POINT_EEG``.
         coord_frame : int
             The coordinate frame used, e.g. ``FIFFV_COORD_HEAD``.
    """

    def __init__(self, info=None, ch_names=None, ch_types=None):
        if all(arg is None for arg in (info, ch_names, ch_types)):
            raise RuntimeError(
                "Either 'info' or 'ch_names' and 'ch_types' must not be None.")
        elif info is None and all(arg is not None
                                  for arg in (ch_names, ch_types)):
            _check_type(ch_names, (None, 'int', str, list, tuple),  'ch_names')
            _check_type(ch_types, (None, str, list, tuple), 'ch_types')
            self._init_from_channels(ch_names, ch_types)
        elif info is not None and all(arg is None
                                      for arg in (ch_names, ch_types)):
            _check_type(info, (None, Info), 'info')
            self._init_from_info(info)
        else:
            raise RuntimeError(
                "If 'info' is provided, 'ch_names' and 'ch_types' must be "
                "None. If 'ch_names' and 'ch_types' are provided, 'info' "
                "must be None.")

    def _init_from_info(self, info):
        """Init instance from mne Info."""
        self['custom_ref_applied'] = info['custom_ref_applied']
        self['bads'] = info['bads']
        self['chs'] = tuple(info['chs'])
        self['dig'] = None if info['dig'] is None else tuple(info['dig'])
        self._update_redundant()

    def _init_from_channels(self, ch_names, ch_types):
        """Init instance from channel names and types."""
        # convert ch_names to immutable
        if isinstance(ch_names, _IntLike()):
            ch_names = np.arange(ch_names).astype(str)
        ch_names = tuple(ch_names)

        # retrieve number of channels
        nchan = len(ch_names)

        # convert ch_types to immutable
        if isinstance(ch_types, str):
            ch_types = [ch_types] * nchan
        ch_types = tuple(ch_types)

        # check shape of ch_types
        if np.atleast_1d(np.array(ch_types, np.str_)).ndim != 1 or \
            len(ch_types) != nchan:
            raise ValueError(
                'ch_types and ch_names must be the same length '
                f'({len(ch_types)} != {nchan}) for ch_types={ch_types}')

        # add custom ref flag
        self['custom_ref_applied'] = FIFF.FIFFV_MNE_CUSTOM_REF_OFF

        # init empty bads
        self['bads'] = list()  # mutable, as this can be changed by the user

        # create chs information
        self['chs'] = list()
        ch_types_dict = get_channel_type_constants(include_defaults=True)
        for ci, (ch_name, ch_type) in enumerate(zip(ch_names, ch_types)):
            _check_type(ch_name, (str, ))
            _check_type(ch_type, (str, ))
            if ch_type not in ch_types_dict:
                raise KeyError(
                    f'kind must be one of {list(ch_types_dict)}, not '
                    f'{ch_type}.')
            this_ch_dict = ch_types_dict[ch_type]
            kind = this_ch_dict['kind']
            # handle chpi, where kind is a *list* of FIFF constants:
            kind = kind[0] if isinstance(kind, (list, tuple)) else kind
            # mirror what tag.py does here
            coord_frame = _ch_coord_dict.get(kind, FIFF.FIFFV_COORD_UNKNOWN)
            coil_type = this_ch_dict.get('coil_type', FIFF.FIFFV_COIL_NONE)
            unit = this_ch_dict.get('unit', FIFF.FIFF_UNIT_NONE)
            chan_info = dict(loc=np.full(12, np.nan),
                             unit_mul=FIFF.FIFF_UNITM_NONE, range=1., cal=1.,
                             kind=kind, coil_type=coil_type, unit=unit,
                             coord_frame=coord_frame, ch_name=str(ch_name),
                             scanno=ci + 1, logno=ci + 1)
            self['chs'].append(chan_info)

        # convert to immutable
        self['chs'] = tuple(self['chs'])

        # add empty dig
        self['dig'] = None

        # add ch_names and nchan
        self._update_redundant()

    def _update_redundant(self):
        """Update the redundant entries."""
        self['ch_names'] = tuple([ch['ch_name'] for ch in self['chs']])
        self['nchan'] = len(self['chs'])

    def copy(self):
        """Copy the instance.

        Returns
        -------
        info : instance of ChInfo
            The copied info.
        """
        return deepcopy(self)

    # ------------------------------------------------------------------------
    @property
    def bads(self):
        """List of bad channels."""
        return self['bads']

    @property
    def chs(self):
        """Channel information dictionary."""
        return self['chs']

    @property
    def ch_names(self):
        """Tuple of the channel names."""
        return self['ch_names']

    @property
    def dig(self):
        """Montage"""
        return self['dig']

    @property
    def nchan(self):
        """Number of channels."""
        return self['nchan']

    @property
    def custom_ref_applied(self):
        """The reference applied."""
        # TODO: This should be replaced by knowledge of what the reference
        # actually is. see mne-python/issues/8962
        return self['custom_ref_applied']
