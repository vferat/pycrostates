import numpy as np
import scipy
from numpy.typing import NDArray

from ..._typing import Cluster
from ...utils._checks import _check_type
from ...utils._docs import fill_doc


def _optimize_order(
    centers: NDArray[float],
    template_centers: NDArray[float],
    ignore_polarity: bool = True,
):
    n_states = len(centers)
    M = np.corrcoef(template_centers, centers)[:n_states, n_states:]
    if ignore_polarity:
        M = np.abs(M)
    _, order = scipy.optimize.linear_sum_assignment(-M)
    return order


@fill_doc
def optimize_order(inst: Cluster, template_inst: Cluster):
    """Optimize the order of cluster centers between two cluster instances.

    Optimize the order of cluster centers in an instance of a clustering
    algorithm to maximize auto-correlation, based on a template instance
    as determined by the Hungarian algorithm.
    The two cluster instances must have the same number of cluster centers
    and the same polarity setting.

    Parameters
    ----------
    inst : :ref:`cluster`
        Fitted clustering algorithm to reorder.
    template_inst : :ref:`cluster`
        Fitted clustering algorithm to use as template for reordering.

    Returns
    -------
    order : list of int
        The new order to apply to inst to maximize auto-correlation
        of cluster centers.
    """
    from .._base import _BaseCluster

    _check_type(inst, (_BaseCluster,), item_name="inst")
    inst._check_fit()
    _check_type(template_inst, (_BaseCluster,), item_name="template_inst")
    template_inst._check_fit()

    if inst.n_clusters != template_inst.n_clusters:
        raise ValueError(
            "Instance and the template must have the same "
            "number of cluster centers."
        )
    if inst._ignore_polarity != template_inst._ignore_polarity:
        raise ValueError(
            "Cannot find order: Instance was fitted using "
            f"ignore_polarity={inst._ignore_polarity} while "
            "template was fitted using ignore_polarity="
            f"{inst._ignore_polarity} which could lead to "
            "misinterpretations."
        )
    inst_centers = inst._cluster_centers_
    template_centers = template_inst._cluster_centers_
    order = _optimize_order(
        inst_centers, template_centers, ignore_polarity=inst._ignore_polarity
    )
    return order
