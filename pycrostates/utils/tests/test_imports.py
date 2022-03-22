"""Test _imports.py"""

import pytest

from pycrostates.utils._logs import logger, set_log_level
from pycrostates.utils._imports import import_optional_dependency

set_log_level('INFO')
logger.propagate = True


def test_import_optional_dependency(caplog):
    """Test the import of optional dependencies."""
    # Test import of present package
    numpy = import_optional_dependency('numpy')
    assert isinstance(numpy.__version__, str)

    # Test import of absent package
    caplog.clear()
    with pytest.raises(ImportError, match="Missing optional dependency"):
        import_optional_dependency('non_existing_pkg', raise_error=True)
    assert "Missing optional dependency" not in caplog.text

    # Test import of absent package without raise
    caplog.clear()
    pkg = import_optional_dependency('non_existing_pkg', raise_error=False)
    assert pkg is None
    assert "Missing optional dependency" in caplog.text

    # Test extra
    with pytest.raises(ImportError, match="blabla"):
        import_optional_dependency('non_existing_pkg', extra='blabla')
