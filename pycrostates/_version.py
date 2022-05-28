"""Version number."""

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # for python 3.7

__version__ = version(__package__)
