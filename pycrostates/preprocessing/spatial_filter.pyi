from _typeshed import Incomplete
from mne import BaseEpochs
from mne.io import BaseRaw
from scipy.sparse import csr_matrix

from .._typing import ScalarFloatArray as ScalarFloatArray
from ..io import ChData as ChData
from ..utils._checks import _check_n_jobs as _check_n_jobs
from ..utils._checks import _check_type as _check_type
from ..utils._checks import _check_value as _check_value
from ..utils._docs import fill_doc as fill_doc
from ..utils._logs import logger as logger

def _check_adjacency(adjacency, info, ch_type):
    """Check adjacency matrix."""

def apply_spatial_filter(
    inst: BaseRaw | BaseEpochs | ChData,
    ch_type: str = "eeg",
    exclude_bads: bool = True,
    origin: str | ScalarFloatArray = "auto",
    adjacency: csr_matrix | str = "auto",
    n_jobs: int = 1,
    verbose: Incomplete | None = None,
):
    """Apply a spatial filter.

    Adapted from \\ :footcite:t:`michel2019eeg`. Apply an instantaneous filter which
    interpolates channels with local neighbors while removing outliers.
    The current implementation proceeds as follows:

    * An interpolation matrix is computed using
      ``mne.channels.interpolation._make_interpolation_matrix``.
    * An ajdacency matrix is computed using `mne.channels.find_ch_adjacency`.
    * If ``exclude_bads`` is set to ``True``, bad channels are removed from the
      ajdacency matrix.
    * For each timepoint and each channel, a reduced adjacency matrix is built by
      removing neighbors with lowest and highest value.
    * For each timepoint and each channel, a reduced interpolation matrix is built by
      extracting neighbor weights based on the reduced adjacency matrix.
    * The reduced interpolation matrices are normalized.
    * The channel's timepoints are interpolated using their reduced interpolation
      matrix.

    Parameters
    ----------
    inst : Raw | Epochs | ChData
        Instance to filter spatially.
    ch_type : str
        The channel type on which to apply the spatial filter. Currently only supports
        ``'eeg'``.
    exclude_bads : bool
        If set to ``True``, bad channels will be removed from the adjacency matrix and
        therefore not used to interpolate neighbors. In addition, bad channels will not
        be filtered. If set to ``False``, proceed as if all channels were good.
    origin : array of shape (3,) | str
        Origin of the sphere in the head coordinate frame and in meters. Can be
        ``'auto'`` (default), which means a head-digitization-based origin fit.
    adjacency : array or csr_matrix of shape (n_channels, n_channels) | str
        An adjacency matrix. Can be created using :func:`mne.channels.find_ch_adjacency`
        and edited with :func:`mne.viz.plot_ch_adjacency`. If ``'auto'`` (default), the
        matrix will be automatically created using
        :func:`mne.channels.find_ch_adjacency` and other parameters.
    n_jobs : int | None
        The number of jobs to run in parallel. If ``-1``, it is set
        to the number of CPU cores. Requires the :mod:`joblib` package.
        ``None`` (default) is a marker for 'unset' that will be interpreted
        as ``n_jobs=1`` (sequential execution) unless the call is performed under
        a :class:`joblib:joblib.parallel_config` context manager that sets another
        value for ``n_jobs``.
    verbose : int | str | bool | None
        Sets the verbosity level. The verbosity increases gradually between ``"CRITICAL"``,
        ``"ERROR"``, ``"WARNING"``, ``"INFO"`` and ``"DEBUG"``. If None is provided, the
        verbosity is set to ``"WARNING"``. If a bool is provided, the verbosity is set to
        ``"WARNING"`` for False and to ``"INFO"`` for True.

    Returns
    -------
    inst : Raw | Epochs| ChData
        The instance modified in place.

    Notes
    -----
    This function requires a full copy of the data in memory.

    References
    ----------
    .. footbibliography::
    """

def _channel_spatial_filter(index, data, adjacency_vector, interpolate_matrix): ...
