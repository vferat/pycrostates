import logging
import sys
from typing import Any, Callable, TypeVar

import mne
from decorator import FunctionMaker

from ._checks import _check_type
from ._docs import fill_doc

logger = logging.getLogger(__package__.split(".utils", maxsplit=1)[0])
logger.propagate = False  # don't propagate (in case of multiple imports)


def init_logger(verbose="INFO"):
    """
    Initialize a logger.

    Assign sys.stdout as a handler of the logger.

    Parameters
    ----------
    verbose : int | str
        Logger verbosity.
    """
    set_log_level(verbose)
    add_stream_handler(sys.stdout, verbose)


def add_stream_handler(stream, verbose="INFO"):
    """
    Add a stream handler to the logger.

    The handler redirects the logger output to the stream.

    Parameters
    ----------
    stream : The output stream, e.g. sys.stdout
    verbose : int | str
        Handler verbosity.
    """
    handler = logging.StreamHandler(stream)
    handler.setFormatter(LoggerFormatter())
    logger.addHandler(handler)

    _check_type(verbose, (bool, str, int, None), item_name="verbose")
    if verbose is None:
        verbose = "INFO"
    set_handler_log_level(verbose, -1)


def add_file_handler(fname, mode="a", verbose="INFO"):
    """
    Add a file handler to the logger.

    The handler saves the logs to file.

    Parameters
    ----------
    fname : str | Path
    mode : str
        Mode in which the file is opened.
    verbose : int | str
        Handler verbosity.
    """
    handler = logging.FileHandler(fname, mode)
    handler.setFormatter(LoggerFormatter())
    logger.addHandler(handler)

    _check_type(verbose, (bool, str, int, None), item_name="verbose")
    if verbose is None:
        verbose = "INFO"
    set_handler_log_level(verbose, -1)


def set_handler_log_level(verbose, handler_id=0):
    """
    Set the log level for a specific handler.

    First handler (ID 0) is always stdout, followed by user-defined handlers.

    Parameters
    ----------
    verbose : int | str
        Logger verbosity.
    handler_id : int
        ID of the handler among 'logger.handlers'.
    """
    _check_type(verbose, (bool, str, int, None), item_name="verbose")
    if verbose is None:
        verbose = "INFO"
    logger.handlers[handler_id].setLevel = verbose


def set_log_level(verbose, return_old_level=False):
    """
    Set the log level for the logger.

    Parameters
    ----------
    verbose : int | str
        Logger verbosity.
    """
    _check_type(verbose, (bool, str, int, None), item_name="verbose")
    old_verbose = logger.level
    if verbose is None:
        verbose = "INFO"
    logger.setLevel(verbose)
    return old_verbose if return_old_level else None


class LoggerFormatter(logging.Formatter):
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


# Provide help for static type checkers:
# https://mypy.readthedocs.io/en/stable/generics.html#declaring-decorators
_FuncT = TypeVar("_FuncT", bound=Callable[..., Any])


def verbose(function: _FuncT) -> _FuncT:
    """Verbose decorator to allow functions to override log-level.

    Parameters
    ----------
    function : callable
        Function to be decorated by setting the verbosity level.

    Returns
    -------
    dec : callable
        The decorated function.

    Notes
    -----
    This decorator is used to set the verbose level during a function or method
    call, such as :func:`mne.compute_covariance`. The `verbose` keyword
    argument can be 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', True (an
    alias for 'INFO'), or False (an alias for 'WARNING'). To set the global
    verbosity level for all functions, use :func:`mne.set_log_level`.
    This function also serves as a docstring filler.
    """  # noqa: E501
    # See https://decorator.readthedocs.io/en/latest/tests.documentation.html
    # #dealing-with-third-party-decorators
    try:
        fill_doc(function)
    except TypeError:  # nothing to add
        pass

    # Anything using verbose should either have `verbose=None` in the signature
    # or have a `self.verbose` attribute (if in a method). This code path
    # will raise an error if neither is the case.
    body = """\
def %(name)s(%(signature)s):\n
    try:
        verbose
    except UnboundLocalError:
        try:
            verbose = self.verbose
        except NameError:
            raise RuntimeError('Function %%s does not accept verbose parameter'
                               %% (_function_,))
        except AttributeError:
            raise RuntimeError('Method %%s class does not have self.verbose'
                               %% (_function_,))
    else:
        if verbose is None:
            try:
                verbose = self.verbose
            except (NameError, AttributeError):
                pass
    if verbose is not None:
        with _use_log_level_(verbose):
            return _function_(%(shortsignature)s)
    else:
        return _function_(%(shortsignature)s)"""
    evaldict = dict(_use_log_level_=use_log_level, _function_=function)
    fm = FunctionMaker(function, None, None, None, None, function.__module__)
    attrs = dict(
        __wrapped__=function,
        __qualname__=function.__qualname__,
        __globals__=function.__globals__,
    )
    return fm.make(body, evaldict, addsource=True, **attrs)


class use_log_level:
    """Context handler for logging level.

    Parameters
    ----------
    level : int
        The level to use.
    """

    def __init__(self, level):  # noqa: D102
        self.level = level

    def __enter__(self):  # noqa: D105
        self.old_level = set_log_level(self.level, return_old_level=True)

    def __exit__(self, *args):  # noqa: D105
        set_log_level(self.old_level)


def _set_verbose(verbose):
    """
    Similar to verbose decorator.

    Parameters
    ----------
    verbose : int | str
        Logger verbosity.
    """
    mne.set_log_level(verbose)
    set_log_level(verbose)


init_logger()
