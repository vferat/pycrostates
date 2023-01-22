from typing import Union

import numpy as np
from mne import BaseEpochs, pick_info
from mne.bem import _check_origin
from mne.channels import find_ch_adjacency
from mne.channels.interpolation import _make_interpolation_matrix
from mne.io import BaseRaw
from mne.io.pick import _picks_by_type
from mne.parallel import parallel_func
from mne.utils.check import _check_preload
from numpy.typing import NDArray

from .._typing import CHData
from ..utils._checks import _check_n_jobs, _check_type, _check_value
from ..utils._docs import fill_doc
from ..utils._logs import logger, verbose


@fill_doc
@verbose
def apply_spatial_filter(
    inst: Union[BaseRaw, BaseEpochs, CHData],
    ch_type: str = "eeg",
    exclude_bads: bool = True,
    origin: Union[str, NDArray[float]] = "auto",
    n_jobs: int = 1,
    verbose=None,
):
    r"""Apply a spatial filter.

    Adapted from \ :footcite:t:`michel2019eeg`.
    Apply an instantaneous filter which interpolates channels
    with local neighbors while removing outliers.
    The current implementation proceeds as follows:

    * An interpolation matrix is computed using
      ``mne.channels.interpolation._make_interpolation_matrix``.
    * An ajdacency matrix is computed using
      `mne.channels.find_ch_adjacency`.
    * If ``exclude_bads`` is set to ``True``,
      bad channels are removed from the ajdacency matrix.
    * For each timepoint and each channel,
      a reduced adjacency matrix is built by removing neighbors
      with lowest and highest value.
    * For each timepoint and each channel,
      a reduced interpolation matrix is built by extracting neighbor
      weights based on the reduced adjacency matrix.
    * The reduced interpolation matrices are normalized.
    * The channel's timepoints are interpolated
      using their reduced interpolation matrix.

    Parameters
    ----------
    inst : Raw | Epochs | ChData
        Instance to filter spatially.
    ch_type : str
        The channel type on which to apply the spatial filter.
        Currently only supports ``'eeg'``.
    exclude_bads : bool
        If set to ``True``, bad channels will be removed
        from the adjacency matrix and therefore not used
        to interpolate neighbors. In addition, bad channels
        will not be filtered.
        If set to ``False``, proceed as if all channels were good.
    origin : array of shape (3,) | str
        Origin of the sphere in the head coordinate frame and in meters.
        Can be ``'auto'`` (default), which means a head-digitization-based
        origin fit.
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
    """
    _check_type(inst, (BaseRaw, BaseEpochs, CHData), item_name="inst")
    _check_type(ch_type, (str,), item_name="ch_type")
    _check_value(ch_type, ("eeg",), item_name="ch_type")
    _check_type(exclude_bads, (bool,), item_name="exclude_bads")
    origin = _check_origin(origin, inst.info)
    n_jobs = _check_n_jobs(n_jobs)
    _check_preload(inst, "Apply spatial filter")
    if inst.get_montage() is None:
        raise ValueError(
            "No montage was set on your data, but spatial filter"
            "can only work if digitization points for the EEG "
            "channels are available. Consider calling inst.set_montage() "
            "to apply a montage."
        )
    # extract adjacency matrix
    adjacency_matrix, ch_names = find_ch_adjacency(inst.info, ch_type)
    adjacency_matrix = adjacency_matrix.todense()
    adjacency_matrix = np.array(adjacency_matrix)
    if exclude_bads:
        for c, chan in enumerate(ch_names):
            if chan in inst.info["bads"]:
                adjacency_matrix[c, :] = 0  # do not change bads
                adjacency_matrix[:, c] = 0  # don't use bads to interpolate
    # retrieve picks based on adjacency matrix
    picks = dict(_picks_by_type(inst.info, exclude=[]))[ch_type]
    info = pick_info(inst.info, picks)
    assert ch_names == info.ch_names
    # retrieve channel positions
    pos = inst._get_channel_positions(picks)
    # test spherical fit
    origin = _check_origin(origin, info)
    distance = np.linalg.norm(pos - origin, axis=-1)
    distance = np.mean(distance / np.mean(distance))
    if np.abs(1.0 - distance) > 0.1:
        logger.warn(
            "Your spherical fit is poor, interpolation results are "
            "likely to be inaccurate."
        )
    pos = pos - origin
    interpolate_matrix = _make_interpolation_matrix(pos, pos)
    # retrieve data
    data = inst.get_data(picks=picks)
    if isinstance(inst, BaseEpochs):
        data = np.hstack(data)
    # apply filter
    if n_jobs == 1:
        spatial_filtered_data = []
        for index, adjacency_vector in enumerate(adjacency_matrix):
            channel_data = _channel_spatial_filter(
                index, data, adjacency_vector, interpolate_matrix
            )
            spatial_filtered_data.append(channel_data)
    else:
        parallel, p_fun, _ = parallel_func(
            _channel_spatial_filter, n_jobs, total=len(adjacency_matrix)
        )
        spatial_filtered_data = parallel(
            p_fun(index, data, adjacency_vector, interpolate_matrix)
            for index, adjacency_vector in enumerate(adjacency_matrix)
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
    neighbor_matrix = np.array(
        [neighbor_indices.flatten().tolist()] * data.shape[-1]
    ).T

    # Create a mask
    max_mask = neighbors_data == np.amax(neighbors_data, keepdims=True, axis=0)
    min_mask = neighbors_data == np.amin(neighbors_data, keepdims=True, axis=0)
    keep_mask = ~(max_mask | min_mask)

    keep_indices = np.array(
        [
            neighbor_matrix[:, i][keep_mask[:, i]]
            for i in range(keep_mask.shape[-1])
        ]
    )
    channel_data = data[index]
    for i, keep_ind in enumerate(keep_indices):
        weights = interpolate_matrix[keep_ind, index]
        # normalize weights
        weights = weights / np.linalg.norm(weights)
        # average
        channel_data[i] = np.average(data[keep_ind, i], weights=weights)
    return channel_data
