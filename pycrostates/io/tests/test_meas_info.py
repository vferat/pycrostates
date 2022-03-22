"""Test _imports.py"""

import pytest

from pycrostates.io import ChInfo
from pycrostates.utils._logs import logger, set_log_level

set_log_level('INFO')
logger.propagate = True
