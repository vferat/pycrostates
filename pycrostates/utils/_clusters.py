import numpy as np
import scipy

from ..cluster._base import _BaseCluster
from ._checks import _check_type
from ._docs import fill_doc


def _optimize_order(centers, template_centers, ignore_polarity=True):
    n_states = len(centers)
    M = np.corrcoef(template_centers, centers)[:n_states, n_states:]
    if ignore_polarity:
        M = np.abs(M)
    _, order = scipy.optimize.linear_sum_assignment(-M)
    return order


@fill_doc
def optimize_order(inst, template_inst):
    """Find order that best match instance and template cluster centers.

    Compute the optimal assignment of indices in instance cluster centers
    to indices of template cluster centers, to maximize clusters
    auto-correlation as determined by the Hungarian algorithm.

    Parameters
    ----------
    %(cluster)s

    Returns
    -------
    order : list of int
        The new order to apply to inst to maximize auto-correlation
        of cluster centers.
    """
    _check_type(inst, (_BaseCluster,), item_name="inst")
    inst._check_fit()
    _check_type(template_inst, (_BaseCluster,), item_name="template_inst")
    template_inst._check_fit()

    if inst.n_clusters != template_inst.n_clusters:
        raise ValueError(
            "Instance and the template must have the same "
            "number of cluster centers"
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
