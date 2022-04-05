"""Test _checks.py"""

import os
import re

from matplotlib import pyplot as plt
from numpy import isclose
from numpy.random import Generator, PCG64
from numpy.random.mtrand import RandomState
import pytest

from pycrostates.utils._checks import (
    _ensure_int, _check_type, _check_value, _check_n_jobs, _check_random_state,
    _check_axes)


def test_ensure_int():
    """Test _ensure_int checker."""
    assert _ensure_int(101) == 101
    with pytest.raises(TypeError, match='Item must be an int'):
        _ensure_int(101.)
    with pytest.raises(TypeError, match='Item must be an int'):
        _ensure_int(True)
    with pytest.raises(TypeError, match='Item must be an int'):
        _ensure_int([101])


def test_check_type():
    """Test _check_type checker."""
    # valids
    assert _check_type(101, ('int', )) == 101
    assert _check_type('101.fif', ('path-like', )) == '101.fif'

    def foo():
        pass
    _check_type(foo, ('callable', ))

    assert _check_type(101, ('numeric', )) == 101
    assert _check_type(101., ('numeric', )) == 101.

    # invalids
    with pytest.raises(TypeError, match="Item must be an instance of"):
        _check_type(101, (float, ))
    with pytest.raises(TypeError, match="'number' must be an instance of"):
        _check_type(101, (float, ), 'number')


def test_check_value():
    """Test _check_value checker."""
    # valids
    assert _check_value(5, [1, 2, 3, 4, 5]) == 5
    assert _check_value((1, 2), [(1, 2), (2, 3, 4, 5)]) == (1, 2)

    # invalids
    with pytest.raises(ValueError, match="Invalid value for the parameter."):
        _check_value(5, [1, 2, 3, 4])
    with pytest.raises(ValueError,
                       match="Invalid value for the 'number' parameter."):
        _check_value(5, [1, 2, 3, 4], 'number')


def test_check_n_jobs():
    """Test _check_n_jobs checker."""
    assert _check_n_jobs(1) == 1
    assert _check_n_jobs(5) == 5
    assert _check_n_jobs(101) == 101

    with pytest.raises(ValueError, match="n_jobs must be an integer"):
        _check_n_jobs('cuda')

    # all cores
    n = os.cpu_count()
    for k in range(n):
        assert _check_n_jobs(-k-1) == n - k
    assert _check_n_jobs(-n) == 1
    with pytest.raises(ValueError, match="If n_jobs has a negative value"):
        _check_n_jobs(-n-1)


def test_check_random_state():
    """Test _check_random_state checker."""
    rs = _check_random_state(None)
    assert isinstance(rs, RandomState)
    rs2 = _check_random_state(101)
    assert isinstance(rs, RandomState)
    rs3 = _check_random_state(102)
    assert isinstance(rs, RandomState)
    assert not isclose(rs2.normal(), rs3.normal())

    rng = Generator(PCG64())
    rs = _check_random_state(rng)
    assert isinstance(rs, Generator)

    with pytest.raises(ValueError,
                       match=re.escape("[101] cannot be used to seed")):
        _check_random_state([101])


def test_check_ax():
    """Test _check_ax checker."""
    # test valid inputs
    _, ax = plt.subplots(1, 1)
    _check_axes(ax)
    plt.close('all')
    _, ax = plt.subplots(1, 2)
    _check_axes(ax)
    plt.close('all')
    _, ax = plt.subplots(2, 1)
    _check_axes(ax)
    plt.close('all')

    # test invalid inputs
    f, ax = plt.subplots(1, 1)
    with pytest.raises(TypeError, match="must be an instance of"):
        _check_axes(f)
    plt.close('all')
    _, ax = plt.subplots(10, 10)
    ax = ax.reshape((2, 5, 10))
    with pytest.raises(ValueError, match="Argument 'ax' should be a"):
        _check_axes(ax)
    plt.close('all')
