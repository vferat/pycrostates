"""Test _docs.py"""

from pycrostates.utils._docs import copy_doc, fill_doc


def test_fill_doc():
    """Test decorator to fill docstring."""

    @fill_doc
    def foo(n_clusters, verbose):
        """My doc.

        Parameters
        ----------
        %(n_clusters)s
        %(verbose)s
        """
        pass

    assert "n_clusters : int" in foo.__doc__
    assert "verbose : bool" in foo.__doc__


def test_copy_doc():
    """Test decorator to copy docstring."""

    def foo(x, y):
        """
        My doc.
        """
        pass

    @copy_doc(foo)
    def foo2(x, y):
        pass

    assert "My doc." in foo2.__doc__
