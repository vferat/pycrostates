from collections.abc import Callable
from typing import IO

from packaging.requirements import Requirement

from ._checks import _check_type as _check_type

def sys_info(fid: IO | None = None, *, extra: bool = False, developer: bool = False):
    """Print the system information for debugging.

    Parameters
    ----------
    fid : file-like | None
        The file to write to, passed to :func:`print`. Can be None to use
        :data:`sys.stdout`.
    extra : bool
        If True, display information about optional dependencies.
    developer : bool
        If True, display information about developer dependencies. Only available for
        the package installed in editable mode.
    """

def _list_dependencies_info(
    out: Callable, ljust: int, package: str, dependencies: list[Requirement]
):
    """List dependencies names and versions."""
