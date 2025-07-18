import logging
from collections.abc import Callable
from functools import wraps
from pathlib import Path

import mne

from ._checks import _check_type, _check_verbose
from ._docs import fill_doc
from ._fixes import _WrapStdOut


@fill_doc
def _init_logger(*, verbose: bool | str | int | None = None) -> logging.Logger:
    """Initialize a logger.

    Assigns sys.stdout as the first handler of the logger.

    Parameters
    ----------
    %(verbose)s

    Returns
    -------
    logger : Logger
        The initialized logger.
    """
    # create logger
    verbose = _check_verbose(verbose)
    logger = logging.getLogger(__package__.split(".utils", maxsplit=1)[0])
    logger.propagate = False
    logger.setLevel(verbose)

    # add the main handler
    handler = logging.StreamHandler(_WrapStdOut())
    handler.setFormatter(_LoggerFormatter())
    logger.addHandler(handler)

    return logger


@fill_doc
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
    %(verbose)s
    """
    verbose = _check_verbose(verbose)
    handler = logging.FileHandler(fname, mode, encoding)
    handler.setFormatter(_LoggerFormatter())
    handler.setLevel(verbose)
    logger.addHandler(handler)


@fill_doc
def set_log_level(verbose: bool | str | int | None, apply_to_mne: bool = True) -> None:
    """Set the log level for the logger and the first handler ``sys.stdout``.

    Parameters
    ----------
    %(verbose)s
    apply_to_mne : bool
        If True, also changes the log level of MNE.
    """
    _check_type(apply_to_mne, (bool,), "apply_to_mne")
    verbose = _check_verbose(verbose)
    if apply_to_mne:
        mne.set_log_level(verbose)
    logger.setLevel(verbose)


class _LoggerFormatter(logging.Formatter):
    """Format string Syntax for pycrostates."""

    # Format string syntax for the different Log levels
    _formatters = {}
    _formatters[logging.DEBUG] = logging.Formatter(
        fmt="[%(module)s:%(funcName)s:%(lineno)d] %(levelname)s: %(message)s "
        "(%(asctime)s)"
    )
    _formatters[logging.INFO] = logging.Formatter(
        fmt="[%(module)s.%(funcName)s] %(levelname)s: %(message)s"
    )
    _formatters[logging.WARNING] = logging.Formatter(
        fmt="[%(module)s.%(funcName)s] %(levelname)s: %(message)s"
    )
    _formatters[logging.ERROR] = logging.Formatter(
        fmt="[%(module)s:%(funcName)s:%(lineno)d] %(levelname)s: %(message)s"
    )

    def __init__(self):
        super().__init__(fmt="%(levelname): %(message)s")

    def format(self, record):
        """
        Format the received log record.

        Parameters
        ----------
        record : logging.LogRecord
        """
        if record.levelno <= logging.DEBUG:
            return self._formatters[logging.DEBUG].format(record)
        if record.levelno <= logging.INFO:
            return self._formatters[logging.INFO].format(record)
        if record.levelno <= logging.WARNING:
            return self._formatters[logging.WARNING].format(record)
        return self._formatters[logging.ERROR].format(record)


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

    @wraps(f)
    def wrapper(*args, **kwargs):
        if "verbose" in kwargs:
            with _use_log_level(kwargs["verbose"]):
                return f(*args, **kwargs)
        else:
            return f(*args, **kwargs)

    return wrapper


@fill_doc
class _use_log_level:
    """Context manager to change the logging level temporary.

    Parameters
    ----------
    %(verbose)s
    """

    def __init__(self, verbose: bool | str | int | None = None):
        self._old_level = logger.level
        self._level = verbose

    def __enter__(self):
        mne.set_log_level(self._level)
        set_log_level(self._level)

    def __exit__(self, *args):
        mne.set_log_level(self._old_level)
        set_log_level(self._old_level)


logger = _init_logger(verbose="WARNING")  # equivalent to verbose=None
