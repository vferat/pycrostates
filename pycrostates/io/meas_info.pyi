from mne import Info

from ..utils._checks import _check_type as _check_type
from ..utils._checks import _IntLike as _IntLike
from ..utils._logs import logger as logger

class ChInfo(Info):
    """Atemporal measurement information.

    Similar to a :class:`mne.Info` class, but without any temporal information.
    Only the channel-related information are present. A :class:`~pycrostates.io.ChInfo`
    can be created either:

    - by providing a :class:`~mne.Info` class from which information are retrieved.
    - by providing the ``ch_names`` and the ``ch_types`` to create a new instance.

    Only one of those 2 methods should be used at once.

    .. warning:: The only entry that should be manually changed by the user is
                 ``info['bads']``. All other entries should be considered read-only,
                 though they can be modified by various functions or methods (which have
                 safeguards to ensure all fields remain in sync).

    Parameters
    ----------
    info : Info | None
        MNE measurement information instance from which channel-related variables are
        retrieved.
    ch_names : list of str | int | None
        Channel names. If an int, a list of channel names will be created from
        ``range(ch_names)``.
    ch_types : list of str | str | None
        Channel types. If str, all channels are assumed to be of the same type.

    Attributes
    ----------
    bads : list of str
        List of bad (noisy/broken) channels, by name. These channels will by default be
        ignored by many processing steps.
    ch_names : tuple of str
        The names of the channels.
    chs : tuple of dict
        A list of channel information dictionaries, one per channel. See Notes for more
        information.
    comps : list of dict
        CTF software gradient compensation data. See Notes for more information.
    ctf_head_t : dict | None
        The transformation from 4D/CTF head coordinates to Neuromag head coordinates.
        This is only present in 4D/CTF data.
    custom_ref_applied : int
        Whether a custom (=other than average) reference has been applied to the EEG
        data. This flag is checked by some algorithms that require an average reference
        to be set.
    dev_ctf_t : dict | None
        The transformation from device coordinates to 4D/CTF head coordinates. This is
        only present in 4D/CTF data.
    dev_head_t : dict | None
        The device to head transformation.
    dig : tuple of dict | None
        The Polhemus digitization data in head coordinates. See Notes for more
        information.
    nchan : int
        Number of channels.
    projs : list of Projection
        List of SSP operators that operate on the data. See :class:`mne.Projection` for
        details.

    Notes
    -----
    The following parameters have a nested structure.

    * ``chs`` list of dict:

        cal : float
            The calibration factor to bring the channels to physical units. Used in
            product with ``range`` to scale the data read from disk.
        ch_name : str
            The channel name.
        coil_type : int
            Coil type, e.g. ``FIFFV_COIL_MEG``.
        coord_frame : int
            The coordinate frame used, e.g. ``FIFFV_COORD_HEAD``.
        kind : int
            The kind of channel, e.g. ``FIFFV_EEG_CH``.
        loc : array, shape (12,)
            Channel location. For MEG this is the position plus the normal given by a
            3x3 rotation matrix. For EEG this is the position followed by reference
            position (with 6 unused). The values are specified in device coordinates for
            MEG and in head coordinates for EEG channels, respectively.
        logno : int
            Logical channel number, conventions in the usage of this number vary.
        range : float
            The hardware-oriented part of the calibration factor. This should be only
            applied to the continuous raw data. Used in product with ``cal`` to scale
            data read from disk.
        scanno : int
            Scanning order number, starting from 1.
        unit : int
            The unit to use, e.g. ``FIFF_UNIT_T_M``.
        unit_mul : int
            Unit multipliers, most commonly ``FIFF_UNITM_NONE``.

    * ``comps`` list of dict:

        ctfkind : int
            CTF compensation grade.
        colcals : ndarray
            Column calibrations.
        mat : dict
            A named matrix dictionary (with entries "data", "col_names", etc.)
            containing the compensation matrix.
        rowcals : ndarray
            Row calibrations.
        save_calibrated : bool
            Were the compensation data saved in calibrated form.

    * ``dig`` list of dict:

        kind : int
            The kind of channel, e.g. ``FIFFV_POINT_EEG``, ``FIFFV_POINT_CARDINAL``.
        r : array, shape (3,)
            3D position in m. and coord_frame.
        ident : int
            Number specifying the identity of the point. e.g. ``FIFFV_POINT_NASION`` if
            kind is ``FIFFV_POINT_CARDINAL``, or 42 if kind is ``FIFFV_POINT_EEG``.
        coord_frame : int
            The coordinate frame used, e.g. ``FIFFV_COORD_HEAD``.
    """

    def __init__(
        self,
        info: Info | None = None,
        ch_names: int | list[str] | tuple[str, ...] | None = None,
        ch_types: str | list[str] | tuple[str, ...] | None = None,
    ) -> None: ...
    def _init_from_info(self, info: Info):
        """Init instance from mne Info."""
    _unlocked: bool

    def _init_from_channels(
        self,
        ch_names: int | list[str] | tuple[str, ...],
        ch_types: str | list[str] | tuple[str, ...],
    ):
        """Init instance from channel names and types."""

    def __getattribute__(self, name):
        """Attribute getter."""

    def __eq__(self, other):
        """Equality == method."""

    def __ne__(self, other):
        """Different != method."""

    def __deepcopy__(self, memodict):
        """Make a deepcopy."""

    def _check_consistency(self, prepend_error: str = ""):
        """Do some self-consistency checks and datatype tweaks."""
