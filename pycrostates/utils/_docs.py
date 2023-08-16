"""
Fill docstrings to avoid redundant docstrings in multiple files.

Inspired from mne: https://mne.tools/stable/index.html
Inspired from mne.utils.docs.py by Eric Larson <larson.eric.d@gmail.com>
"""
import sys
from typing import Callable, Dict, List, Tuple

from mne.utils.docs import docdict as docdict_mne

# ------------------------- Documentation dictionary -------------------------
docdict: Dict[str, str] = {}

# ---- Documentation to inc. from MNE ----
keys: Tuple[str, ...] = (
    "n_jobs",
    "picks_all",
    "random_state",
    "tmin_raw",
    "tmax_raw",
    "reject_by_annotation_raw",
)

for key in keys:
    entry = docdict_mne[key]
    if ".. versionchanged::" in entry:
        entry = entry.replace(".. versionchanged::", ".. versionchanged:: MNE ")
    if ".. versionadded::" in entry:
        entry = entry.replace(".. versionadded::", ".. versionadded:: MNE ")
    docdict[key] = entry

docdict[
    "verbose"
] = """
verbose : int | str | bool | None
    Sets the verbosity level. The verbosity increases gradually between ``"CRITICAL"``,
    ``"ERROR"``, ``"WARNING"``, ``"INFO"`` and ``"DEBUG"``. If None is provided, the
    verbosity is set to ``"WARNING"``. If a bool is provided, the verbosity is set to
    ``"WARNING"`` for False and to ``"INFO"`` for True."""

# ---- Clusters ----
docdict[
    "n_clusters"
] = """
n_clusters : int
    The number of clusters, i.e. the number of microstates.
"""
docdict[
    "cluster_centers"
] = """
cluster_centers : array (n_clusters, n_channels)
    Fitted clusters, i.e. the microstates maps."""
docdict[
    "cluster_names"
] = """
cluster_names : list | None
    Name of the clusters."""

# ---- Metrics -----
docdict[
    "cluster"
] = """
cluster : :ref:`cluster`
    Fitted clustering algorithm from which to compute score. For more details about
    current clustering implementations, check the :ref:`Clustering` section of the
    documentation.
"""

# ------ I/O -------
docdict[
    "fname_fiff"
] = """
fname : str | Path
    Path to the ``.fif`` file where the clustering solution is saved."""

# -- Segmentation --
docdict[
    "cluster_centers_seg"
] = """
cluster_centers : array (n_clusters, n_channels)
     Clusters, i.e. the microstates maps used to compute the segmentation."""
docdict[
    "labels_raw"
] = """
labels : array of shape ``(n_samples,)``
    Microstates labels attributed to each sample, i.e. the segmentation."""
docdict[
    "labels_epo"
] = """
labels : array of shape ``(n_epochs, n_samples)``
    Microstates labels attributed to each sample, i.e. the segmentation."""
docdict[
    "labels_transition"
] = """
labels : array of shape ``(n_samples,)`` or ``(n_epochs, n_samples)``
    Microstates labels attributed to each sample, i.e. the segmentation."""
# TODO: predict_parameters docstring is missing.
docdict[
    "predict_parameters"
] = """
predict_parameters : dict | None
    The prediction parameters."""
docdict[
    "stat_transition"
] = """
stat : str
    Aggregate statistic to compute transitions. Can be:

    * ``count``: show the number of observations of each transition.
    * ``probability`` or ``proportion``: normalize count such as the probabilities along
      the first axis is always equal to ``1``.
    * ``percent``: normalize count such as the probabilities along the first axis is
      always equal to ``100``."""
docdict[
    "stat_expected_transitions"
] = """
stat : str
    Aggregate statistic to compute transitions. Can be:

    * ``probability`` or ``proportion``: normalize count such as the probabilities along
      the first axis is always equal to ``1``.
    * ``percent``: normalize count such as the probabilities along the first axis is
      always equal to ``100``."""
docdict[
    "ignore_self"
] = """
ignore_self : bool
    If True, ignores the transition from one state to itself. This is equivalent to
    setting the duration of all states to 1 sample."""
docdict[
    "transition_matrix"
] = """
T : array of shape ``(n_cluster, n_cluster)``
    Array of transition probability values from one label to another.
    First axis indicates state ``"from"``. Second axis indicates state ``"to"``."""

# ------ Viz -------
docdict[
    "cmap"
] = """
cmap : str | colormap | None
    The colormap to use. If None, ``viridis`` is used."""
docdict[
    "block"
] = """
block : bool
    Whether to halt program execution until the figure is closed."""
docdict[
    "axes_topo"
] = """
axes : Axes | None
    Either ``None`` to create a new figure or axes (or an array of axes) on which the
    topographic map should be plotted. If the number of microstates maps to plot is
    ``≥ 1``, an array of axes of size ``n_clusters`` should be provided."""
docdict[
    "axes_seg"
] = """
axes : Axes | None
    Either ``None`` to create a new figure or axes on which the segmentation is
    plotted."""
docdict[
    "axes_cbar"
] = """
cbar_axes : Axes | None
    Axes on which to draw the colorbar, otherwise the colormap takes space from the main
    axes."""

# ------------------------- Documentation functions --------------------------
docdict_indented: Dict[int, Dict[str, str]] = {}


def fill_doc(f: Callable) -> Callable:
    """Fill a docstring with docdict entries.

    Parameters
    ----------
    f : callable
        The function to fill the docstring of (modified in place).

    Returns
    -------
    f : callable
        The function, potentially with an updated __doc__.
    """
    docstring = f.__doc__
    if not docstring:
        return f

    lines = docstring.splitlines()
    indent_count = _indentcount_lines(lines)

    try:
        indented = docdict_indented[indent_count]
    except KeyError:
        indent = " " * indent_count
        docdict_indented[indent_count] = indented = dict()

        for name, docstr in docdict.items():
            lines = [
                indent + line if k != 0 else line
                for k, line in enumerate(docstr.strip().splitlines())
            ]
            indented[name] = "\n".join(lines)

    try:
        f.__doc__ = docstring % indented
    except (TypeError, ValueError, KeyError) as exp:
        funcname = f.__name__
        funcname = docstring.split("\n")[0] if funcname is None else funcname
        raise RuntimeError(f"Error documenting {funcname}:\n{str(exp)}")

    return f


def _indentcount_lines(lines: List[str]) -> int:
    """Minimum indent for all lines in line list.

    >>> lines = [' one', '  two', '   three']
    >>> indentcount_lines(lines)
    1
    >>> lines = []
    >>> indentcount_lines(lines)
    0
    >>> lines = [' one']
    >>> indentcount_lines(lines)
    1
    >>> indentcount_lines(['    '])
    0
    """
    indent = sys.maxsize
    for k, line in enumerate(lines):
        if k == 0:
            continue
        line_stripped = line.lstrip()
        if line_stripped:
            indent = min(indent, len(line) - len(line_stripped))
    if indent == sys.maxsize:
        return 0
    return indent


def copy_doc(source: Callable) -> Callable:
    """Copy the docstring from another function (decorator).

    The docstring of the source function is prepepended to the docstring of the
    function wrapped by this decorator.

    This is useful when inheriting from a class and overloading a method. This
    decorator can be used to copy the docstring of the original method.

    Parameters
    ----------
    source : callable
        The function to copy the docstring from.

    Returns
    -------
    wrapper : callable
        The decorated function.

    Examples
    --------
    >>> class A:
    ...     def m1():
    ...         '''Docstring for m1'''
    ...         pass
    >>> class B(A):
    ...     @copy_doc(A.m1)
    ...     def m1():
    ...         ''' this gets appended'''
    ...         pass
    >>> print(B.m1.__doc__)
    Docstring for m1 this gets appended
    """

    def wrapper(func):
        if source.__doc__ is None or len(source.__doc__) == 0:
            raise RuntimeError(
                f"The docstring from {source.__name__} could not be copied "
                "because it was empty."
            )
        doc = source.__doc__
        if func.__doc__ is not None:
            doc += func.__doc__
        func.__doc__ = doc
        return func

    return wrapper
