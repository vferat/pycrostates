from abc import ABC
from typing import Optional, Union

from numpy.random import Generator, RandomState
from numpy.typing import NDArray

class CHData(ABC):
    """Typing for CHData."""

class CHInfo(ABC):
    """Typing for CHInfo."""

class Cluster(ABC):
    """Typing for a clustering class."""

class Segmentation(ABC):
    """Typing for a clustering class."""
RANDomState = Optional[Union[int, RandomState, Generator]]
Picks = Optional[Union[str, NDArray[int]]]