import os
import warnings

import pytest
from mne import set_log_level as set_log_level_mne

from pycrostates import set_log_level
from pycrostates.utils._logs import logger


def pytest_configure(config):
    """Configure pytest options."""
    config.addinivalue_line("usefixtures", "matplotlib_config")

    warnings_lines = r"""
    error::
    # We use matplotlib agg backend to avoid any window to pop up during tests.
    ignore:Matplotlib is currently using agg:UserWarning
    # Pytest internals
    ignore:Use setlocale.*instead:DeprecationWarning
    ignore:datetime\.datetime\.utcnow.*is deprecated.*:DeprecationWarning
    """
    for warning_line in warnings_lines.split("\n"):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith("#"):
            config.addinivalue_line("filterwarnings", warning_line)

    logger.propagate = True
    set_log_level_mne("WARNING")
    set_log_level("WARNING")


@pytest.fixture(scope="session")
def matplotlib_config():
    """Configure matplotlib for viz tests."""
    import matplotlib
    from matplotlib import cbook

    # Allow for easy interactive debugging with a call like:
    #
    #     $ PYCROSTATES_MPL_TESTING_BACKEND=Qt5Agg pytest mne/viz/tests/test_raw.py -k annotation -x --pdb  # noqa: E501
    #
    try:
        want = os.environ["PYCROSTATES_MPL_TESTING_BACKEND"]
    except KeyError:
        want = "agg"  # don't pop up windows
    with warnings.catch_warnings(record=True):  # ignore warning
        warnings.filterwarnings("ignore")
        matplotlib.use(want, force=True)
    import matplotlib.pyplot as plt

    assert plt.get_backend() == want
    # overwrite some params that can horribly slow down tests that
    # users might have changed locally (but should not otherwise affect
    # functionality)
    plt.ioff()
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["figure.raise_window"] = False

    # Make sure that we always reraise exceptions in handlers
    orig = cbook.CallbackRegistry

    class CallbackRegistryReraise(orig):
        def __init__(self, exception_handler=None, signals=None):
            super(CallbackRegistryReraise, self).__init__(exception_handler)

    cbook.CallbackRegistry = CallbackRegistryReraise


@pytest.fixture(autouse=True)
def close_all():
    """Close all matplotlib plots, regardless of test status."""
    # This adds < 1 µS in local testing
    import matplotlib.pyplot as plt

    yield
    plt.close("all")
