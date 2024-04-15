from pathlib import Path

from ..utils._checks import _check_type as _check_type
from ..utils._docs import fill_doc as fill_doc
from ..utils._logs import logger as logger

def read_cluster(fname: str | Path):
    """Read clustering solution from disk.

    Parameters
    ----------
    fname : str | Path
        Path to the ``.fif`` file where the clustering solution is saved.

    Returns
    -------
    cluster : :ref:`Clustering`
        Fitted clustering instance.
    """
