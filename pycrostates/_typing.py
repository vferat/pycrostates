from __future__ import annotations  # c.f. PEP 563, PEP 649

from abc import ABC
from typing import Generic, Optional, TypeVar, Union

import numpy as np
from matplotlib.axes import Axes
from numpy.random import Generator, RandomState

ScalarFloatType = TypeVar("ScalarFloatType", np.float32, np.float64)
ScalarIntType = TypeVar("ScalarIntType", np.int8, np.int16, np.int32, np.int64)
ScalarType = TypeVar(
    "ScalarType", np.int8, np.int16, np.int32, np.int64, np.float32, np.float64
)


class ScalarFloatArray(np.ndarray, Generic[ScalarFloatType]):
    pass


class ScalarIntArray(np.ndarray, Generic[ScalarIntType]):
    pass


class ScalarArray(np.ndarray, Generic[ScalarType]):
    pass


class AxesArray(np.ndarray, Generic[TypeVar("AxesType", bound=Axes)]):
    pass


class Segmentation(ABC):  # noqa: B024
    """Typing for a clustering class."""

    pass


RANDomState = Optional[Union[int, RandomState, Generator]]
Picks = Optional[Union[str, ScalarIntArray]]
