"""Utils module for utilities."""

from . import sys_info  # noqa: F401
from ._config import get_config, set_config
from .utils import _compare_infos, _corr_vectors, _distance_matrix  # noqa: F401

__all__: tuple[str, ...] = ("get_config", "set_config")
