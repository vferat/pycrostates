"""
Fill docstrings to avoid redundant docstrings in multiple files.
Inspired from mne: https://mne.tools/stable/index.html
Inspired from mne.utils.docs.py by Eric Larson <larson.eric.d@gmail.com>
"""
import sys

from mne.utils.docs import docdict as docdict_mne

# ------------------------- Documentation dictionary -------------------------
docdict = dict()

# ---- Documentation to inc. from MNE ----
keys = ['random_state', 'verbose',
        'reject_by_annotation_raw', 'n_jobs',
        'tmin_raw', 'tmax_raw', 'picks_all']

for key in keys:
    docdict[key] = docdict_mne[key]
# TODO: sphinx :term:`data channels` in 'picks_all' has to be included.

# ---- Clusters ----
docdict['fit_inst'] = """
inst : `~mne.io.Raw` | `~mne.Epochs`
    Instance containing data to transform to cluster-distance space
    (absolute spatial correlation)."""
docdict['predict_inst'] = """
inst : `~mne.io.Raw` | `~mne.Epochs`
    Instance containing data to predict."""

docdict['n_clusters'] = """
n_clusters : int
    The number of clusters as well as the number
    of centroids (i.e. Microstate topographies).
"""

docdict['random_seed'] = """
random_seed : float
    As estimation can be non-deterministic it can be useful to fix the
    random state to have reproducible results.
"""

docdict['verbose'] = """
verbose : bool, str, int or None
    Control verbosity of the logging output.
    If None, use the default verbosity level.
    Should only be passed as a keyword argument.
    Defaults to self.verbose.
"""

docdict['picks'] = """
picks : str | list | slice | None
    Channels to include. Slices and lists of integers will be interpreted
    as channel indices. In lists, channel type strings (e.g.,
    ['meg', 'eeg']) will pick channels of those types, channel name strings
    (e.g., ['MEG0111', 'MEG2623'] will pick the given channels.
    Can also be the string values “all” to pick all channels, or “data” to
    pick data channels. None will pick all channels.
    Note that channels in info['bads'] will be included.
    Default to 'eeg'.
"""

# ------------------------- Documentation functions --------------------------
docdict_indented = dict()


def fill_doc(f):
    """Fill a docstring with docdict entries.
    Parameters
    ----------
    f : callable
        The function to fill the docstring of. Will be modified in place.
    Returns
    -------
    f : callable
        The function, potentially with an updated ``__doc__``.
    """
    docstring = f.__doc__
    if not docstring:
        return f
    lines = docstring.splitlines()
    # Find the minimum indent of the main docstring, after first line
    if len(lines) < 2:
        icount = 0
    else:
        icount = _indentcount_lines(lines[1:])
    # Insert this indent to dictionary docstrings
    try:
        indented = docdict_indented[icount]
    except KeyError:
        indent = ' ' * icount
        docdict_indented[icount] = indented = {}
        for name, dstr in docdict.items():
            lines = dstr.splitlines()
            try:
                newlines = [lines[0]]
                for line in lines[1:]:
                    newlines.append(indent + line)
                indented[name] = '\n'.join(newlines)
            except IndexError:
                indented[name] = dstr
    try:
        f.__doc__ = docstring % indented
    except (TypeError, ValueError, KeyError) as exp:
        funcname = f.__name__
        funcname = docstring.split('\n')[0] if funcname is None else funcname
        raise RuntimeError('Error documenting %s:\n%s'
                           % (funcname, str(exp)))
    return f


def _indentcount_lines(lines):
    """Compute minimum indent for all lines in line list."""
    indentno = sys.maxsize
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            indentno = min(indentno, len(line) - len(stripped))
    if indentno == sys.maxsize:
        return 0
    return indentno


def copy_doc(source):
    """
    Copy the docstring from another function (decorator).

    The docstring of the source function is prepepended to the docstring of the
    function wrapped by this decorator.

    This is useful when inheriting from a class and overloading a method. This
    decorator can be used to copy the docstring of the original method.

    Parameters
    ----------
    source : function
        Function to copy the docstring from.

    Returns
    -------
    wrapper : function
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
            raise ValueError('Cannot copy docstring: docstring was empty.')
        doc = source.__doc__
        if func.__doc__ is not None:
            doc += func.__doc__
        func.__doc__ = doc
        return func
    return wrapper
