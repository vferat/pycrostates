from __future__ import annotations  # c.f. PEP 563, PEP 649

from typing import TYPE_CHECKING

import numpy as np
from mne import BaseEpochs, pick_info
from mne.bem import _check_origin
from mne.channels import find_ch_adjacency
from mne.channels.interpolation import _make_interpolation_matrix
from mne.io import BaseRaw
from mne.parallel import parallel_func
from mne.utils import check_version
from mne.utils.check import _check_preload
from scipy.sparse import csr_array, csr_matrix

if check_version("mne", "1.6"):
    from mne._fiff.pick import _picks_by_type
else:
    from mne.io.pick import _picks_by_type

from ..utils._checks import _check_n_jobs, _check_type, _check_value
from ..utils._docs import fill_doc
from ..utils._logs import logger, verbose

if TYPE_CHECKING:
    from typing import Union

    from .._typing import ScalarFloatArray
    from ..io import ChData


def _check_adjacency(adjacency, info, ch_type):
    """Check adjacency matrix."""
    # in MNE 1.8, the type was changed from a csr_matrix to a csr_array
    _check_type(adjacency, (csr_matrix, csr_array, np.ndarray, str), "adjacency")
    # auto
    if isinstance(adjacency, str):
        if adjacency != "auto":
            raise (
                ValueError(
                    "Adjacency can be either a scipy.sparse.csr_array (MNE 1.8 "
                    "and above), a scipy.sparse.csr_matrix (MNE 1.7 and older) or "
                    f"'auto' but got string '{adjacency}' instead."
                )
            )
        adjacency, ch_names = find_ch_adjacency(info, ch_type)
    # custom
    if not isinstance(adjacency, np.ndarray):
        adjacency = adjacency.toarray()
    ch_names = info.ch_names
    n_channels = len(ch_names)
    if adjacency.ndim != 2:
        raise ValueError(
            f"Adjacency must have exactly 2 dimensions but got {adjacency.ndim} "
            "dimensions instead."
        )
    if (adjacency.shape[0] != n_channels) or (adjacency.shape[1] != n_channels):
        raise ValueError(
            "Adjacency must be of shape (n_channels, n_channels) but got "
            f"{adjacency.shape} instead."
        )
    if not np.array_equal(adjacency, adjacency.astype(bool)):
        raise ValueError("Values contained in adjacency can only be 0 or 1.")
    return (adjacency, ch_names)


@fill_doc
@verbose
def apply_spatial_filter(
    inst: Union[BaseRaw, BaseEpochs, ChData],
    ch_type: str = "eeg",
    exclude_bads: bool = True,
    origin: Union[str, ScalarFloatArray] = "auto",
    adjacency: Union[csr_matrix, str] = "auto",
    n_jobs: int = 1,
    verbose=None,
):
    r"""Apply a spatial filter.

    Adapted from \ :footcite:t:`michel2019eeg`. Apply an instantaneous filter which
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
    %(n_jobs)s
    %(verbose)s

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
    """  # noqa: E501
    from ..io import ChData

    _check_type(inst, (BaseRaw, BaseEpochs, ChData), item_name="inst")
    _check_type(ch_type, (str,), item_name="ch_type")
    _check_value(ch_type, ("eeg",), item_name="ch_type")
    _check_type(exclude_bads, (bool,), item_name="exclude_bads")
    n_jobs = _check_n_jobs(n_jobs)
    _check_preload(inst, "Apply spatial filter")
    if inst.get_montage() is None:
        raise ValueError(
            "No montage was set on your data, but spatial filter can only work if "
            "digitization points for the EEG channels are available. Consider calling "
            "inst.set_montage() to apply a montage."
        )
    # retrieve picks
    picks = dict(_picks_by_type(inst.info, exclude=[]))[ch_type]
    info = pick_info(inst.info, picks)
    # adjacency matrix
    adjacency, ch_names = _check_adjacency(adjacency, info, ch_type)
    if exclude_bads:
        for c, chan in enumerate(ch_names):
            if chan in inst.info["bads"]:
                adjacency[c, :] = 0  # do not change bads
                adjacency[:, c] = 0  # don't use bads to interpolate
    # retrieve channel positions
    pos = inst._get_channel_positions(picks)
    # test spherical fit
    origin = _check_origin(origin, info)
    distance = np.linalg.norm(pos - origin, axis=-1)
    distance = np.mean(distance / np.mean(distance))
    if np.abs(1.0 - distance) > 0.1:
        logger.warn(
            "Your spherical fit is poor, interpolation results are likely to be "
            "inaccurate."
        )
    pos = pos - origin
    interpolate_matrix = _make_interpolation_matrix(pos, pos)
    # retrieve data
    data = inst.get_data(picks=picks)
    if isinstance(inst, BaseEpochs):
        data = np.hstack(data)
    # apply filter
    logger.info(f"Applying spatial filter on {len(picks)} channels.")
    if n_jobs == 1:
        spatial_filtered_data = []
        for index, adjacency_vector in enumerate(adjacency):
            channel_data = _channel_spatial_filter(
                index, data, adjacency_vector, interpolate_matrix
            )
            spatial_filtered_data.append(channel_data)
    else:
        parallel, p_fun, _ = parallel_func(
            _channel_spatial_filter, n_jobs, total=len(adjacency)
        )
        spatial_filtered_data = parallel(
            p_fun(index, data, adjacency_vector, interpolate_matrix)
            for index, adjacency_vector in enumerate(adjacency)
        )

    data = np.array(spatial_filtered_data)
    if isinstance(inst, BaseEpochs):
        data = data.reshape(
            (len(picks), inst._data.shape[0], inst._data.shape[-1])
        ).swapaxes(0, 1)
        inst._data[:, picks, :] = data
    else:
        inst._data[picks] = data

    return inst


def _channel_spatial_filter(index, data, adjacency_vector, interpolate_matrix):
    neighbors_data = data[adjacency_vector == 1, :]
    neighbor_indices = np.argwhere(adjacency_vector == 1)
    # too much bads /edge
    if len(neighbor_indices) <= 3:
        print(index)
        return data[index]
    # neighbor_matrix shape (n_neighbor, n_samples)
    neighbor_matrix = np.array([neighbor_indices.flatten().tolist()] * data.shape[-1]).T

    # Create a mask
    max_mask = neighbors_data == np.amax(neighbors_data, keepdims=True, axis=0)
    min_mask = neighbors_data == np.amin(neighbors_data, keepdims=True, axis=0)
    keep_mask = ~(max_mask | min_mask)

    keep_indices = np.array(
        [neighbor_matrix[:, i][keep_mask[:, i]] for i in range(keep_mask.shape[-1])]
    )
    channel_data = data[index]
    for i, keep_ind in enumerate(keep_indices):
        weights = interpolate_matrix[keep_ind, index]
        # normalize weights
        weights = weights / np.linalg.norm(weights)
        # average
        channel_data[i] = np.average(data[keep_ind, i], weights=weights)
    return channel_data
