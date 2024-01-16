from numpy.typing import NDArray

from ..._typing import Cluster as Cluster
from ...utils._checks import _check_type as _check_type
from ...utils._docs import fill_doc as fill_doc

def _optimize_order(centers: NDArray[float], template_centers: NDArray[float], ignore_polarity: bool=True):
    ...

def optimize_order(inst: Cluster, template_inst: Cluster):
    """Optimize the order of cluster centers between two cluster instances.

    Optimize the order of cluster centers in an instance of a clustering algorithm to
    maximize auto-correlation, based on a template instance as determined by the
    Hungarian algorithm. The two cluster instances must have the same number of cluster
    centers and the same polarity setting.

    Parameters
    ----------
    inst : :ref:`cluster`
        Fitted clustering algorithm to reorder.
    template_inst : :ref:`cluster`
        Fitted clustering algorithm to use as template for reordering.

    Returns
    -------
    order : list of int
        The new order to apply to inst to maximize auto-correlation of cluster centers.
    """