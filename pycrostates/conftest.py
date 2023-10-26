import os

from mne import set_log_level as set_log_level_mne

from pycrostates import set_log_level


def pytest_configure(config):
    """Configure pytest options."""
    if "MPLBACKEND" not in os.environ:
        os.environ["MPLBACKEND"] = "agg"

    warnings_lines = r"""
    error::
    # We use matplotlib agg backend to avoid any window to pop up during tests.
    ignore:Matplotlib is currently using agg:UserWarning
    ignore:datetime\.utcfromtimestamp.*is deprecated:DeprecationWarning
    """
    for warning_line in warnings_lines.split("\n"):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith("#"):
            config.addinivalue_line("filterwarnings", warning_line)

    set_log_level_mne("WARNING")
    set_log_level("WARNING")
