"""Measurement information, similar to `mne.Info` instance."""

from copy import deepcopy
from numbers import Number
from typing import List, Optional, Tuple, Union

import numpy as np
from mne.io import Info
from mne.io.constants import FIFF
from mne.io.meas_info import _check_bads, _check_ch_keys, _unique_channel_names
from mne.io.pick import get_channel_type_constants
from mne.io.proj import Projection
from mne.io.tag import _ch_coord_dict

from .._typing import CHInfo
from ..utils._checks import _check_type, _IntLike
from ..utils._logs import logger


class ChInfo(CHInfo, Info):
    """Atemporal measurement information.

    Similar to a mne.Info class, but without any temporal information. Only the
    channel-related information are present. A ChInfo can be created either:
        - by providing an `~mne.Info` class from which information are
          retrieved.
        - by providing the ``ch_names`` and the ``ch_types`` to create a new
          instance.
    Only one of those 2 methods should be used at once.

    .. warning:: The only entry that should be manually changed by the user
                 is ``info['bads']``. All other entries should be
                 considered read-only, though they can be modified by various
                 functions or methods (which have safeguards to ensure all
                 fields remain in sync).

    Parameters
    ----------
    info : mne.Info | None
        MNE measurement information instance from which channel-related
        variables are retrieved.
    ch_names : list of str | int | None
        Channel names. If an int, a list of channel names will be created
        from ``range(ch_names)``.
    ch_types : list of str | str | None
        Channel types. If str, all channels are assumed to be of the same type.

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
    comps : list of dict
        CTF software gradient compensation data.
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
    projs : list of Projection
        List of SSP operators that operate on the data.
        See :class:`mne.Projection` for details.

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

    # valid items
    # fmt: off
    _attributes = {
        "bads": _check_bads,
        "ch_names": "ch_names cannot be set directly. "
                    "Please use methods inst.add_channels(), "
                    "inst.drop_channels(), inst.pick_channels(), "
                    "inst.rename_channels(), inst.reorder_channels() "
                    "and inst.set_channel_types() instead.",
        "chs": "chs cannot be set directly. "
               "Please use methods inst.add_channels(), "
               "inst.drop_channels(), inst.pick_channels(), "
               "inst.rename_channels(), inst.reorder_channels() "
               "and inst.set_channel_types() instead.",
        "comps": "comps cannot be set directly. "
                 "Please use method Raw.apply_gradient_compensation() "
                 "instead.",
        "custom_ref_applied": "custom_ref_applied cannot be set directly. "
                              "Please use method inst.set_eeg_reference() "
                              "instead.",
        "dig": "dig cannot be set directly. "
               "Please use method inst.set_montage() instead.",
        "nchan": "nchan cannot be set directly. "
                 "Please use methods inst.add_channels(), "
                 "inst.drop_channels(), and inst.pick_channels() instead.",
        "projs": "projs cannot be set directly. "
                 "Please use methods inst.add_proj() and inst.del_proj() "
                 "instead.",
    }
    # fmt: on

    def __init__(
        self,
        info: Optional[Info] = None,
        ch_names: Optional[
            Union[
                int,
                List[str],
                Tuple[str, ...],
            ]
        ] = None,
        ch_types: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
    ):
        if all(arg is None for arg in (info, ch_names, ch_types)):
            raise RuntimeError(
                "Either 'info' or 'ch_names' and 'ch_types' must not be None."
            )
        if info is None and all(
            arg is not None for arg in (ch_names, ch_types)
        ):
            _check_type(ch_names, (None, "int", list, tuple), "ch_names")
            _check_type(ch_types, (None, str, list, tuple), "ch_types")
            self._init_from_channels(ch_names, ch_types)
        elif info is not None and all(
            arg is None for arg in (ch_names, ch_types)
        ):
            _check_type(info, (None, Info), "info")
            self._init_from_info(info)
        else:
            raise RuntimeError(
                "If 'info' is provided, 'ch_names' and 'ch_types' must be "
                "None. If 'ch_names' and 'ch_types' are provided, 'info' "
                "must be None."
            )

    def _init_from_info(self, info: Info):
        """Init instance from mne Info."""
        self["custom_ref_applied"] = info["custom_ref_applied"]
        self["bads"] = info["bads"]
        self["chs"] = info["chs"]
        self["dig"] = info["dig"]
        self["comps"] = info["comps"]
        self["projs"] = info["projs"]
        self._update_redundant()

    def _init_from_channels(
        self,
        ch_names: Union[int, List[str], Tuple[str, ...]],
        ch_types: Union[str, List[str], Tuple[str, ...]],
    ):
        """Init instance from channel names and types."""
        self._unlocked = True

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
        ndim = np.atleast_1d(np.array(ch_types, np.str_)).ndim
        if ndim != 1 or len(ch_types) != nchan:
            raise ValueError(
                "ch_types and ch_names must be the same length "
                f"({len(ch_types)} != {nchan}) for ch_types={ch_types}"
            )

        # add custom ref flag
        self["custom_ref_applied"] = FIFF.FIFFV_MNE_CUSTOM_REF_OFF

        # init empty bads
        self["bads"] = []  # mutable, as this can be changed by the user

        # create chs information
        self["chs"] = []
        ch_types_dict = get_channel_type_constants(include_defaults=True)
        for ci, (ch_name, ch_type) in enumerate(zip(ch_names, ch_types)):
            _check_type(ch_name, (str,))
            _check_type(ch_type, (str,))
            if ch_type not in ch_types_dict:
                raise KeyError(
                    f"kind must be one of {list(ch_types_dict)}, not "
                    f"{ch_type}."
                )
            this_ch_dict = ch_types_dict[ch_type]
            kind = this_ch_dict["kind"]
            # handle chpi, where kind is a *list* of FIFF constants:
            kind = kind[0] if isinstance(kind, (list, tuple)) else kind
            # mirror what tag.py does here
            coord_frame = _ch_coord_dict.get(kind, FIFF.FIFFV_COORD_UNKNOWN)
            coil_type = this_ch_dict.get("coil_type", FIFF.FIFFV_COIL_NONE)
            unit = this_ch_dict.get("unit", FIFF.FIFF_UNIT_NONE)
            chan_info = dict(
                loc=np.full(12, np.nan),
                unit_mul=FIFF.FIFF_UNITM_NONE,
                range=1.0,
                cal=1.0,
                kind=kind,
                coil_type=coil_type,
                unit=unit,
                coord_frame=coord_frame,
                ch_name=str(ch_name),
                scanno=ci + 1,
                logno=ci + 1,
            )
            self["chs"].append(chan_info)

        # convert to immutable
        self["chs"] = self["chs"]

        # add empty dig
        self["dig"] = None

        # add empty compensation grades
        self["comps"] = []

        # add empty projs
        self["projs"] = []

        self._unlocked = False

        # add ch_names and nchan
        self._update_redundant()

    def __getattribute__(self, name):
        """Attribute getter."""
        # disable method/attributes that pycrostates does not support
        # invalid attributes
        _inv_attributes = ()
        # invalid methods/properties
        _inv_methods = (
            "pick_channels"  # TODO: Can be removed when req. for MNE = 1.1.0
        )
        if name in _inv_attributes or name in _inv_methods:
            raise AttributeError(
                f"'{self.__class__.__name__}' has not attribute '{name}'"
            )
        return super().__getattribute__(name)

    def __eq__(self, other):
        """Equality == method."""
        if isinstance(other, Info):
            # compare channel names
            if self["ch_names"] != other["ch_names"]:
                return False
            assert self["nchan"] == other["nchan"]  # sanity-check

            # compare channel types
            if self.get_channel_types() != other.get_channel_types():
                return False

            # compare montage
            m1 = self.get_montage()
            m2 = other.get_montage()
            if m1 is None and m2 is not None:
                return False
            if m1 is not None and m2 is None:
                return False
            if m1 is not None and m2 is not None:
                if any(dig not in m2.dig for dig in m1.dig):
                    return False
                if any(dig not in m1.dig for dig in m2.dig):
                    return False

            # compare custom ref
            if self["custom_ref_applied"] != other["custom_ref_applied"]:
                return False

            # TODO: Compare projs and compensation grades.

            if self["bads"] != other["bads"]:
                logger.warning("Both info do not have the same bad channels.")

            return True
        else:
            return False

    def __ne__(self, other):
        """Different != method."""
        return not self.__eq__(other)

    # ------------------------------------------------------------------------
    def __setitem__(self, key, val):
        """Setter for Dictionary item."""
        # During unpickling, the _unlocked attribute has not been set, so
        # let __setstate__ do it later and act unlocked now
        unlocked = getattr(self, "_unlocked", True)
        if key in self._attributes:
            if isinstance(self._attributes[key], str):
                if not unlocked:
                    raise RuntimeError(self._attributes[key])
            else:
                val = self._attributes[key](val)  # attribute checker function
        else:
            raise RuntimeError(
                f"Info does not support setting the key {repr(key)}. "
                "Supported keys are "
                f"{', '.join(repr(k) for k in self._attributes)}"
            )
        super().__setitem__(key, val)  # calls the dict __setitem__

    def __deepcopy__(self, memodict):
        """Make a deepcopy."""
        result = ChInfo.__new__(ChInfo)
        result._unlocked = True
        for k, v in self.items():
            # chs is roughly half the time but most are immutable
            if k == "chs":
                # dict shallow copy is fast, so use it then overwrite
                result[k] = []
                for ch in v:
                    ch = ch.copy()  # shallow
                    ch["loc"] = ch["loc"].copy()
                    result[k].append(ch)
            elif k == "ch_names":
                # we know it's list of str, shallow okay and saves ~100 µs
                result[k] = v.copy()
            else:
                result[k] = deepcopy(v, memodict)
        result._unlocked = False
        return result

    def _check_consistency(self, prepend_error: str = ""):
        """Do some self-consistency checks and datatype tweaks."""
        missing = [bad for bad in self["bads"] if bad not in self["ch_names"]]
        if len(missing) > 0:
            msg = "%sbad channel(s) %s marked do not exist in info"
            raise RuntimeError(
                msg
                % (
                    prepend_error,
                    missing,
                )
            )

        chs = [ch["ch_name"] for ch in self["chs"]]
        if (
            len(self["ch_names"]) != len(chs)
            or any(ch_1 != ch_2 for ch_1, ch_2 in zip(self["ch_names"], chs))
            or self["nchan"] != len(chs)
        ):
            raise RuntimeError(
                f"{prepend_error}info channel name inconsistency detected, "
                "please notify developers."
            )

        for pi, proj in enumerate(self.get("projs", [])):
            _check_type(proj, (Projection,), f'info["projs"][{pi}]')
            for key in ("kind", "active", "desc", "data", "explained_var"):
                if key not in proj:
                    raise RuntimeError(f"Projection incomplete, missing {key}")

        # Ensure info['chs'] has immutable entries (copies much faster)
        for ci, ch in enumerate(self["chs"]):
            _check_ch_keys(ch, ci)
            ch_name = ch["ch_name"]
            if not isinstance(ch_name, str):
                raise TypeError(
                    f'Bad info: info["chs"][{ci}]["ch_name"] is not a string, '
                    f"got type {type(ch_name)}."
                )
            for key in (
                "scanno",
                "logno",
                "kind",
                "range",
                "cal",
                "coil_type",
                "unit",
                "unit_mul",
                "coord_frame",
            ):
                val = ch.get(key, 1)
                if not isinstance(val, Number):
                    raise TypeError(
                        f'Bad info: info["chs"][{ci}][{key}] = {val} is type '
                        f"{type(val)}, must be float or int."
                    )
            loc = ch["loc"]
            if not (isinstance(loc, np.ndarray) and loc.shape == (12,)):
                raise TypeError(
                    f'Bad info: info["chs"][{ci}]["loc"] must be ndarray '
                    f"with 12 elements, got {loc}."
                )

        # make sure channel names are unique
        with self._unlock():
            self["ch_names"] = _unique_channel_names(self["ch_names"])
            for idx, ch_name in enumerate(self["ch_names"]):
                self["chs"][idx]["ch_name"] = ch_name
