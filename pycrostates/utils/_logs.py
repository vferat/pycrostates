import sys
import logging
from pathlib import Path

import mne

from ._checks import _check_type


name = Path(__file__).parent.parent.name
logger = logging.getLogger(name)
logger.propagate = False  # don't propagate (in case of multiple imports)


def init_logger(verbose='INFO'):
    """
    Initialize a logger. Assign sys.stdout as a handler of the logger.

    Parameters
    ----------
    verbose : int | str
        Logger verbosity.
    """
    set_log_level(verbose)
    add_stream_handler(sys.stdout, verbose)


def add_stream_handler(stream, verbose='INFO'):
    """
    Add a handler to the logger. The handler redirects the logger output to
    the stream.

    Parameters
    ----------
    stream : The output stream, e.g. sys.stdout
    verbose : int | str
        Handler verbosity.
    """
    handler = logging.StreamHandler(stream)
    handler.setFormatter(LoggerFormatter())
    logger.addHandler(handler)

    _check_type(verbose, (bool, str, int, None), item_name='verbose')
    if verbose is None:
        verbose = 'INFO'
    set_handler_log_level(verbose, -1)


def add_file_handler(fname, mode='a', verbose='INFO'):
    """
    Add a file handler to the logger. The handler saves the logs to file.

    Parameters
    ----------
    fname : str | Path
    mode : str
        Mode in which the file is openned.
    verbose : int | str
        Handler verbosity.
    """
    handler = logging.FileHandler(fname, mode)
    handler.setFormatter(LoggerFormatter())
    logger.addHandler(handler)

    _check_type(verbose, (bool, str, int, None), item_name='verbose')
    if verbose is None:
        verbose = 'INFO'
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
    _check_type(verbose, (bool, str, int, None), item_name='verbose')
    if verbose is None:
        verbose = 'INFO'
    logger.handlers[handler_id].setLevel = verbose


def set_log_level(verbose):
    """
    Set the log level for the logger.

    Parameters
    ----------
    verbose : int | str
        Logger verbosity.
    """
    _check_type(verbose, (bool, str, int, None), item_name='verbose')
    if verbose is None:
        verbose = 'INFO'
    logger.setLevel(verbose)


class LoggerFormatter(logging.Formatter):
    """
    Format string Syntax for BSL.
    """
    # Format string syntax for the different Log levels
    _formatters = dict()
    _formatters[logging.DEBUG] = logging.Formatter(
        fmt="[%(module)s:%(funcName)s:%(lineno)d] %(levelname)s: %(message)s "
            "(%(asctime)s)")
    _formatters[logging.INFO] = logging.Formatter(
        fmt="[%(module)s.%(funcName)s] %(levelname)s: %(message)s")
    _formatters[logging.WARNING] = logging.Formatter(
        fmt="[%(module)s.%(funcName)s] %(levelname)s: %(message)s")
    _formatters[logging.ERROR] = logging.Formatter(
        fmt="[%(module)s:%(funcName)s:%(lineno)d] %(levelname)s: %(message)s")

    def __init__(self):
        super().__init__(fmt='%(levelname): %(message)s')

    def format(self, record):
        """
        Format the received log record.

        Parameters
        ----------
        record : logging.LogRecord
        """
        if record.levelno <= logging.DEBUG:
            return self._formatters[logging.DEBUG].format(record)
        elif record.levelno <= logging.INFO:
            return self._formatters[logging.INFO].format(record)
        elif record.levelno <= logging.WARNING:
            return self._formatters[logging.WARNING].format(record)
        else:
            return self._formatters[logging.ERROR].format(record)


def verbose(f):
    """
    Set the verbose for MNE and pycrostates.

    Parameters
    ----------
    f : callable
        The function with a verbose argument.

    Returns
    -------
    f : callable
        The function.
    """
    def wrapper(*args, **kwargs):
        if 'verbose' in kwargs:
            mne.set_log_level(kwargs['verbose'])
            set_log_level(kwargs['verbose'])
        return f(*args, **kwargs)
    return wrapper


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
