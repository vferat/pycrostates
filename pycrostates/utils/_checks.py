"""Utility functions for checking types and values. Inspired from MNE."""

import multiprocessing as mp
import operator
import os
from itertools import product
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from mne.utils import check_random_state


def _ensure_int(item, item_name=None):
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
    # This is preferred over numbers.Integral, see:
    # https://github.com/scipy/scipy/pull/7351#issuecomment-299713159
    try:
        # someone passing True/False is much more likely to be an error than
        # intentional usage
        if isinstance(item, bool):
            raise TypeError
        item = int(operator.index(item))
    except TypeError:
        item_name = "Item" if item_name is None else "'%s'" % item_name
        raise TypeError(
            "%s must be an int, got %s instead." % (item_name, type(item))
        )

    return item


class _IntLike:
    @classmethod
    def __instancecheck__(cls, other):
        try:
            _ensure_int(other)
        except TypeError:
            return False
        else:
            return True


class _Callable:
    @classmethod
    def __instancecheck__(cls, other):
        return callable(other)


_types = {
    "numeric": (np.floating, float, _IntLike()),
    "path-like": (str, Path, os.PathLike),
    "int": (_IntLike(),),
    "callable": (_Callable(),),
}


def _check_type(item, types, item_name=None):
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
    check_types = sum(
        (
            (type(None),)
            if type_ is None
            else (type_,)
            if not isinstance(type_, str)
            else _types[type_]
            for type_ in types
        ),
        (),
    )

    if not isinstance(item, check_types):
        type_name = [
            "None"
            if cls_ is None
            else cls_.__name__
            if not isinstance(cls_, str)
            else cls_
            for cls_ in types
        ]
        if len(type_name) == 1:
            type_name = type_name[0]
        elif len(type_name) == 2:
            type_name = " or ".join(type_name)
        else:
            type_name[-1] = "or " + type_name[-1]
            type_name = ", ".join(type_name)
        item_name = "Item" if item_name is None else "'%s'" % item_name
        raise TypeError(
            f"{item_name} must be an instance of {type_name}, "
            f"got {type(item)} instead."
        )

    return item


def _check_value(item, allowed_values, item_name=None, extra=None):
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
        Extra string to append to the invalid value sentence, e.g.
        "when using ico mode".

    Raises
    ------
    ValueError
        When the value of the item is not one of the valid options.
    """
    if item not in allowed_values:
        item_name = "" if item_name is None else " '%s'" % item_name
        extra = "" if extra is None else " " + extra
        msg = (
            "Invalid value for the{item_name} parameter{extra}. "
            "{options}, but got {item!r} instead."
        )
        allowed_values = tuple(allowed_values)  # e.g., if a dict was given
        if len(allowed_values) == 1:
            options = "The only allowed value is %s" % repr(allowed_values[0])
        elif len(allowed_values) == 2:
            options = "Allowed values are %s and %s" % (
                repr(allowed_values[0]),
                repr(allowed_values[1]),
            )
        else:
            options = "Allowed values are "
            options += ", ".join([f"{repr(v)}" for v in allowed_values[:-1]])
            options += f", and {repr(allowed_values[-1])}"
        raise ValueError(
            msg.format(
                item_name=item_name, extra=extra, options=options, item=item
            )
        )

    return item


def _check_n_jobs(n_jobs):
    """
    Check n_jobs parameter.

    Check that n_jobs is a positive integer or a negative integer for all
    cores. CUDA is not supported.
    """
    _check_type(n_jobs, ("int",), "n_jobs")
    if n_jobs <= 0:
        n_cores = mp.cpu_count()
        n_jobs_orig = n_jobs
        n_jobs = min(n_cores + n_jobs + 1, n_cores)
        if n_jobs <= 0:
            raise ValueError(
                f"If n_jobs has a non-positive value ({n_jobs_orig}), it must "
                f"not be less than the number of CPUs present ({n_cores})."
            )
    return n_jobs


def _check_random_state(seed):
    """Turn seed into a numpy.random.mtrand.RandomState instance."""
    return check_random_state(seed)


def _check_axes(axes):
    """Check that ax is an Axes object or an array of Axes."""
    _check_type(axes, (Axes, np.ndarray), "axes")
    if isinstance(axes, np.ndarray):
        if axes.ndim == 1:
            for ax in axes:
                _check_type(ax, (Axes,))
                assert hasattr(ax, "plot")  # sanity-check
        elif axes.ndim == 2:
            for i, j in product(range(axes.shape[0]), range(axes.shape[1])):
                _check_type(axes[i, j], (Axes,))
                assert hasattr(axes[i, j], "plot")  # sanity-check
        else:
            raise ValueError(
                "Argument 'axes' should be a matplotib axes or a "
                "1D or 2D numpy array of matplotlib axes."
            )
    else:
        # sanity-check
        assert hasattr(axes, "plot")
    return axes


def _check_reject_by_annotation(reject_by_annotation: bool) -> bool:
    """Check the reject_by_annotation argument."""
    _check_type(
        reject_by_annotation,
        (bool, str, None),
        item_name="reject_by_annotation",
    )
    if isinstance(reject_by_annotation, bool):
        if reject_by_annotation:
            reject_by_annotation = "omit"
        else:
            reject_by_annotation = None
    elif isinstance(reject_by_annotation, str):
        if reject_by_annotation != "omit":
            raise ValueError(
                "Argument 'reject_by_annotation' only allows for 'False', "
                "'True' (omit), or 'omit'. "
                f"Provided: '{reject_by_annotation}'."
            )
    return reject_by_annotation


def _check_tmin_tmax(inst, tmin, tmax):
    """Check tmin/tmax compared to the provided instance."""
    _check_type(tmin, (None, "numeric"), item_name="tmin")
    _check_type(tmax, (None, "numeric"), item_name="tmax")

    # check positiveness for tmin, tmax
    for name, arg in (("tmin", tmin), ("tmax", tmax)):
        if arg is None:
            continue
        if arg < 0:
            raise ValueError(
                f"Argument '{name}' must be positive. " f"Provided '{arg}'."
            )
    # check tmax is shorter than instance
    if tmax is not None and inst.times[-1] < tmax:
        raise ValueError(
            "Argument 'tmax' must be shorter than the instance "
            f"length. Provided: '{tmax}', larger than "
            f"{inst.times[-1]}s instance."
        )
    # check that tmax is larger than tmin
    if tmax is not None and tmin is not None and tmax <= tmin:
        raise ValueError(
            "Argument 'tmax' must be strictly larger than 'tmin'. "
            f"Provided 'tmin' -> '{tmin}' and 'tmax' -> '{tmax}'."
        )
    # check that tmin is shorter than instance
    if tmin is not None and inst.times[-1] <= tmin:
        raise ValueError(
            "Argument 'tmin' must be shorter than the instance "
            f"length. Provided: '{tmin}', larger than "
            f"{inst.times[-1]}s instance."
        )
    return tmin, tmax
