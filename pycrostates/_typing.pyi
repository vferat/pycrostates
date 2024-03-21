from abc import ABC

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

RANDomState = int | RandomState | Generator | None
Picks = str | NDArray[int] | None
