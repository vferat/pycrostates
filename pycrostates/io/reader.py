from pathlib import Path

from .fiff import read_cluster as read_cluster_fif
from ..utils._checks import _check_type


def read_cluster(fname):
    """Read clustering solution from disk.

    Parameters
    ----------
    fname : path-like
        Path to the .fif file where the clustering solution is saved.
    """
    _check_type(fname, ('path-like', ), 'fname')
    fname = Path(fname)

    readers = {
        '.fif': read_cluster_fif,
        '.fif.gz': read_cluster_fif,
        }

    ext = ''.join(fname.suffixes)
    if ext in readers:
        return readers[ext](fname)
    else:
        raise ValueError("File format is not supported.")
