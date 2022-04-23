from pathlib import Path
from typing import Union

from .fiff import _read_cluster as _read_cluster_fif
from ..utils._checks import _check_type
from ..utils._logs import logger


def read_cluster(fname: Union[str, Path]):
    """Read clustering solution from disk.

    Parameters
    ----------
    fname : path-like
        Path to the .fif file where the clustering solution is saved.
    """
    _check_type(fname, ('path-like', ), 'fname')
    fname = Path(fname)

    readers = {
        '.fif': _read_cluster_fif,
        '.fif.gz': _read_cluster_fif,
        }

    ext = ''.join(fname.suffixes)
    if ext in readers:
        cluster, version = readers[ext](fname)
        logger.info("Cluster solution loaded was saved with pycrostates '%s'.",
                    version)
        return cluster
    else:
        raise ValueError("File format is not supported.")
