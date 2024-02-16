from typing import Any

from _typeshed import Incomplete

from ._docs import fill_doc as fill_doc

def _ensure_int(item, item_name: Incomplete | None = None):
    """
    Ensure a variable is an integer.

    Parameters
    ----------
    item : object
        Item to check.
    item_name : str | None
        Name of the item to show inside the error message.

    Raises
    ------
    TypeError
        When the type of the item is not int.
    """

class _IntLike:
    @classmethod
    def __instancecheck__(cls, other): ...

class _Callable:
    @classmethod
    def __instancecheck__(cls, other): ...

_types: Incomplete

def _check_type(item, types, item_name: Incomplete | None = None):
    """
    Check that item is an instance of types.

    Parameters
    ----------
    item : object
        Item to check.
    types : tuple of types | tuple of str
        Types to be checked against.
        If str, must be one of:
            ('int', 'str', 'numeric', 'path-like', 'callable')
    item_name : str | None
        Name of the item to show inside the error message.

    Raises
    ------
    TypeError
        When the type of the item is not one of the valid options.
    """

def _check_value(
    item,
    allowed_values,
    item_name: Incomplete | None = None,
    extra: Incomplete | None = None,
):
    """
    Check the value of a parameter against a list of valid options.

    Parameters
    ----------
    item : object
        Item to check.
    allowed_values : tuple of objects
        Allowed values to be checked against.
    item_name : str | None
        Name of the item to show inside the error message.
    extra : str | None
        Extra string to append to the invalid value sentence, e.g. "with ico mode".

    Raises
    ------
    ValueError
        When the value of the item is not one of the valid options.
    """

def _check_n_jobs(n_jobs):
    """Check n_jobs parameter.

    Check that n_jobs is a positive integer or a negative integer for all cores. CUDA is
    not supported.
    """

def _check_random_state(seed):
    """Turn seed into a numpy.random.mtrand.RandomState instance."""

def _check_axes(axes):
    """Check that ax is an Axes object or an array of Axes."""

def _check_reject_by_annotation(reject_by_annotation: bool) -> bool:
    """Check the reject_by_annotation argument."""

def _check_tmin_tmax(inst, tmin, tmax):
    """Check tmin/tmax compared to the provided instance."""

def _check_picks_uniqueness(info, picks) -> None:
    """Check that the provided picks yield a single channel type."""

def _check_verbose(verbose: Any) -> int:
    """Check that the value of verbose is valid.

    Parameters
    ----------
    verbose : int | str | bool | None
        Sets the verbosity level. The verbosity increases gradually between ``"CRITICAL"``,
        ``"ERROR"``, ``"WARNING"``, ``"INFO"`` and ``"DEBUG"``. If None is provided, the
        verbosity is set to ``"WARNING"``. If a bool is provided, the verbosity is set to
        ``"WARNING"`` for False and to ``"INFO"`` for True.

    Returns
    -------
    verbose : int
        The verbosity level as an integer.
    """

def _ensure_valid_show(show: Any) -> bool:
    """Check show parameter."""
