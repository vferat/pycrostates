"""Temporary bug-fixes awaiting an upstream fix."""

import sys
from warnings import warn


# https://github.com/sphinx-gallery/sphinx-gallery/issues/1112
class _WrapStdOut(object):
    """Dynamically wrap to sys.stdout.

    This makes packages that monkey-patch sys.stdout (e.g.doctest, sphinx-gallery) work
    properly.
    """

    def __getattr__(self, name):  # noqa: D105
        # Even more ridiculous than this class, this must be sys.stdout (not
        # just stdout) in order for this to work (tested on OSX and Linux).
        if hasattr(sys.stdout, name):
            return getattr(sys.stdout, name)
        else:
            raise AttributeError(f"'file' object has not attribute '{name}'")


def deprecate(old: str, new: str) -> None:
    """Warn about deprecation of an argument."""
    warn(
        f"The '{old}' argument is deprecated and will be removed in future "
        f"versions. Please use '{new}' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
