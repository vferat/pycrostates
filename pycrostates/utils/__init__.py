"""Utils module for utilities."""

from ._config import get_config
from .utils import (  # noqa: F401
    _compare_infos,
    _corr_vectors,
    _distance_matrix,
)

__all__ = ("get_config",)
