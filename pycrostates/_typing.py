"""Typing with ABC class for pycrostates classes.

The type class can be used for type hinting for pycrostates classes that are at
risk of circular imports, or for short-cut types re-grouping different types.
"""

from abc import ABC
from typing import Optional, Union

from numpy.random import Generator, RandomState
from numpy.typing import NDArray


class CHData(ABC):
    """Typing for CHData."""

    pass


class CHInfo(ABC):
    """Typing for CHInfo."""

    pass


RANDomState = Optional[Union[int, RandomState, Generator]]
Picks = Optional[Union[str, NDArray[int]]]
