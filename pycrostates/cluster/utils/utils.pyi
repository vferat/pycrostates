from ..._typing import ScalarFloatArray as ScalarFloatArray
from ...utils._checks import _check_type as _check_type
from ...utils._docs import fill_doc as fill_doc
from .._base import _BaseCluster as _BaseCluster

def _optimize_order(
    centers: ScalarFloatArray,
    template_centers: ScalarFloatArray,
    ignore_polarity: bool = True,
): ...
def optimize_order(inst: _BaseCluster, template_inst: _BaseCluster):
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
