"""Utils module for utilities."""

from . import sys_info  # noqa: F401
from ._config import get_config, set_config
from .utils import (  # noqa: F401
    _compare_infos,
    _correlation,
    _correlation_matrix,
    _distance,
    _distance_matrix,
    _gev,
)

__all__: tuple[str, ...] = ("get_config", "set_config")
