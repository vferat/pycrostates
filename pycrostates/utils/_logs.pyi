import logging
from pathlib import Path as Path
from typing import Callable

from _typeshed import Incomplete

from ._checks import _check_type as _check_type
from ._checks import _check_verbose as _check_verbose
from ._docs import fill_doc as fill_doc
from ._fixes import _WrapStdOut as _WrapStdOut

def _init_logger(*, verbose: bool | str | int | None = None) -> logging.Logger:
    """Initialize a logger.

    Assigns sys.stdout as the first handler of the logger.

    Parameters
    ----------
    verbose : int | str | bool | None
        Sets the verbosity level. The verbosity increases gradually between ``"CRITICAL"``,
        ``"ERROR"``, ``"WARNING"``, ``"INFO"`` and ``"DEBUG"``. If None is provided, the
        verbosity is set to ``"WARNING"``. If a bool is provided, the verbosity is set to
        ``"WARNING"`` for False and to ``"INFO"`` for True.

    Returns
    -------
    logger : Logger
        The initialized logger.
    """

def add_file_handler(
    fname: str | Path,
    mode: str = "a",
    encoding: str | None = None,
    *,
    verbose: bool | str | int | None = None,
) -> None:
    """Add a file handler to the logger.

    Parameters
    ----------
    fname : str | Path
        Path to the file where the logging output is saved.
    mode : str
        Mode in which the file is opened.
    encoding : str | None
        If not None, encoding used to open the file.
    verbose : int | str | bool | None
        Sets the verbosity level. The verbosity increases gradually between ``"CRITICAL"``,
        ``"ERROR"``, ``"WARNING"``, ``"INFO"`` and ``"DEBUG"``. If None is provided, the
        verbosity is set to ``"WARNING"``. If a bool is provided, the verbosity is set to
        ``"WARNING"`` for False and to ``"INFO"`` for True.
    """

def set_log_level(verbose: bool | str | int | None, apply_to_mne: bool = True) -> None:
    """Set the log level for the logger and the first handler ``sys.stdout``.

    Parameters
    ----------
    verbose : int | str | bool | None
        Sets the verbosity level. The verbosity increases gradually between ``"CRITICAL"``,
        ``"ERROR"``, ``"WARNING"``, ``"INFO"`` and ``"DEBUG"``. If None is provided, the
        verbosity is set to ``"WARNING"``. If a bool is provided, the verbosity is set to
        ``"WARNING"`` for False and to ``"INFO"`` for True.
    apply_to_mne : bool
        If True, also changes the log level of MNE.
    """

class _LoggerFormatter(logging.Formatter):
    """Format string Syntax for pycrostates."""

    _formatters: Incomplete

    def __init__(self) -> None: ...
    def format(self, record):
        """
        Format the received log record.

        Parameters
        ----------
        record : logging.LogRecord
        """

def verbose(f: Callable) -> Callable:
    """Set the verbose for the function call from the kwargs.

    Parameters
    ----------
    f : callable
        The function with a verbose argument.

    Returns
    -------
    f : callable
        The function.
    """

class _use_log_level:
    """Context manager to change the logging level temporary.

    Parameters
    ----------
    verbose : int | str | bool | None
        Sets the verbosity level. The verbosity increases gradually between ``"CRITICAL"``,
        ``"ERROR"``, ``"WARNING"``, ``"INFO"`` and ``"DEBUG"``. If None is provided, the
        verbosity is set to ``"WARNING"``. If a bool is provided, the verbosity is set to
        ``"WARNING"`` for False and to ``"INFO"`` for True.
    """

    _old_level: Incomplete
    _level: Incomplete

    def __init__(self, verbose: bool | str | int | None = None) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *args) -> None: ...

logger: Incomplete
