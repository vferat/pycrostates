"""Test _checks.py"""

import os
import re

import numpy as np
import pytest
from matplotlib import pyplot as plt
from mne import EpochsArray, create_info
from mne.io import RawArray
from numpy.random import PCG64, Generator
from numpy.random.mtrand import RandomState

from pycrostates.utils._checks import (
    _check_axes,
    _check_n_jobs,
    _check_random_state,
    _check_reject_by_annotation,
    _check_tmin_tmax,
    _check_type,
    _check_value,
    _ensure_int,
)


def test_ensure_int():
    """Test _ensure_int checker."""
    assert _ensure_int(101) == 101
    with pytest.raises(TypeError, match="Item must be an int"):
        _ensure_int(101.0)
    with pytest.raises(TypeError, match="Item must be an int"):
        _ensure_int(True)
    with pytest.raises(TypeError, match="Item must be an int"):
        _ensure_int([101])


def test_check_type():
    """Test _check_type checker."""
    # valids
    assert _check_type(101, ("int",)) == 101
    assert _check_type("101.fif", ("path-like",)) == "101.fif"

    def foo_function():
        pass

    _check_type(foo_function, ("callable",))

    assert _check_type(101, ("numeric",)) == 101
    assert _check_type(101.0, ("numeric",)) == 101.0

    # invalids
    with pytest.raises(TypeError, match="Item must be an instance of"):
        _check_type(101, (float,))
    with pytest.raises(TypeError, match="'number' must be an instance of"):
        _check_type(101, (float,), "number")


def test_check_value():
    """Test _check_value checker."""
    # valids
    assert _check_value(5, [1, 2, 3, 4, 5]) == 5
    assert _check_value((1, 2), [(1, 2), (2, 3, 4, 5)]) == (1, 2)

    # invalids
    with pytest.raises(ValueError, match="Invalid value for the parameter."):
        _check_value(5, [1, 2, 3, 4])
    with pytest.raises(
        ValueError, match="Invalid value for the 'number' parameter."
    ):
        _check_value(5, [1, 2, 3, 4], "number")


def test_check_n_jobs():
    """Test _check_n_jobs checker."""
    assert _check_n_jobs(1) == 1
    assert _check_n_jobs(5) == 5
    assert _check_n_jobs(101) == 101

    with pytest.raises(TypeError, match="'n_jobs' must be an instance of int"):
        _check_n_jobs("cuda")

    # all cores
    n = os.cpu_count()
    for k in range(n):
        assert _check_n_jobs(-k - 1) == n - k
    assert _check_n_jobs(-n) == 1
    with pytest.raises(ValueError, match="If n_jobs has a non-positive value"):
        _check_n_jobs(-n - 1)


def test_check_random_state():
    """Test _check_random_state checker."""
    rs = _check_random_state(None)
    assert isinstance(rs, RandomState)
    rs2 = _check_random_state(101)
    assert isinstance(rs, RandomState)
    rs3 = _check_random_state(102)
    assert isinstance(rs, RandomState)
    assert not np.isclose(rs2.normal(), rs3.normal())

    rng = Generator(PCG64())
    rs = _check_random_state(rng)
    assert isinstance(rs, Generator)

    with pytest.raises(
        ValueError, match=re.escape("[101] cannot be used to seed")
    ):
        _check_random_state([101])


def test_check_axes():
    """Test _check_axes checker."""
    # test valid inputs
    _, ax = plt.subplots(1, 1)
    _check_axes(ax)
    plt.close("all")
    _, ax = plt.subplots(1, 2)
    _check_axes(ax)
    plt.close("all")
    _, ax = plt.subplots(2, 1)
    _check_axes(ax)
    plt.close("all")

    # test invalid inputs
    f, ax = plt.subplots(1, 1)
    with pytest.raises(TypeError, match="must be an instance of"):
        _check_axes(f)
    plt.close("all")
    _, ax = plt.subplots(10, 10)
    ax = ax.reshape((2, 5, 10))
    with pytest.raises(ValueError, match="Argument 'axes' should be a"):
        _check_axes(ax)
    plt.close("all")


def test_check_reject_by_annotation():
    """Test the checker for reject_by_annoation argument."""
    reject_by_annotation = _check_reject_by_annotation(True)
    assert reject_by_annotation == "omit"
    reject_by_annotation = _check_reject_by_annotation(False)
    assert reject_by_annotation is None
    reject_by_annotation = _check_reject_by_annotation(None)
    assert reject_by_annotation is None

    with pytest.raises(
        TypeError, match="'reject_by_annotation' must be an instance of"
    ):
        _check_reject_by_annotation(1)
    with pytest.raises(
        ValueError, match="'reject_by_annotation' only allows for"
    ):
        _check_reject_by_annotation("101")


def test_check_tmin_tmax():
    """Test the checker for tmin/tmax arguments."""
    # create fake data as 3 sin sources measured across 6 channels
    times = np.linspace(0, 5, 2000)
    signals = np.array([np.sin(2 * np.pi * k * times) for k in (7, 22, 37)])
    coeffs = np.random.rand(6, 3)
    data = np.dot(coeffs, signals) + np.random.normal(
        0, 0.1, (coeffs.shape[0], times.size)
    )
    info = create_info(
        ["Fpz", "Cz", "CPz", "Oz", "M1", "M2"], sfreq=400, ch_types="eeg"
    )
    raw = RawArray(data, info)
    epochs = EpochsArray(data.reshape(5, 6, 400), info)

    # test valid tmin/tmax
    _check_tmin_tmax(raw, None, None)
    _check_tmin_tmax(epochs, None, None)
    _check_tmin_tmax(raw, 1, 4)
    _check_tmin_tmax(epochs, 0, 0.5)

    # test invalid tmin/tmax
    with pytest.raises(
        ValueError, match="Argument 'tmax' must be shorter than"
    ):
        _check_tmin_tmax(raw, 1, 6)
    with pytest.raises(
        ValueError, match="Argument 'tmax' must be shorter than"
    ):
        _check_tmin_tmax(epochs, 1, 6)
    with pytest.raises(ValueError, match="Argument 'tmin' must be positive"):
        _check_tmin_tmax(raw, -1, 4)
    with pytest.raises(ValueError, match="Argument 'tmax' must be positive"):
        _check_tmin_tmax(raw, 1, -4)
    with pytest.raises(
        ValueError, match="Argument 'tmax' must be strictly larger than 'tmin'"
    ):
        _check_tmin_tmax(raw, 3, 1)
    with pytest.raises(
        ValueError, match="Argument 'tmax' must be strictly larger than 'tmin'"
    ):
        _check_tmin_tmax(epochs, 0.3, 0.1)
    with pytest.raises(
        ValueError, match="Argument 'tmin' must be shorter than"
    ):
        _check_tmin_tmax(raw, 6, None)
    with pytest.raises(
        ValueError, match="Argument 'tmin' must be shorter than"
    ):
        _check_tmin_tmax(epochs, 2, None)
