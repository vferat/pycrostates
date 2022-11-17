from typing import List, Optional, Union

import mne
import numpy as np
from mne import BaseEpochs, pick_info
from mne.channels.interpolation import _make_interpolation_matrix
from mne.io import BaseRaw
from mne.io.pick import _picks_by_type, _picks_to_idx
from mne.parallel import parallel_func
from mne.utils.check import _check_preload

from .._typing import CHData, Picks, RANDomState
from ..utils._checks import _check_n_jobs, _check_type, _check_value
from ..utils._docs import fill_doc
from ..utils._logs import logger, verbose


def apply_spatial_filter(
    inst: Union[BaseRaw, BaseEpochs],
    ch_type: str = "eeg",
    exclude_bads: bool = True,
    n_jobs: int = 1,
):  # TODO: add verbose
    # Checks
    _check_type(inst, (BaseRaw, BaseEpochs))
    _check_value(ch_type, ("mag", "grad", "eeg"), item_name="ch_type")
    n_jobs = _check_n_jobs(n_jobs)
    # check preload for Raw
    _check_preload(inst, "Apply spatial filter")
    # remove bad channels
    # TODO: better handling of picks and bad_channels
    if exclude_bads:
        inst = inst.copy().drop_channels(inst.info["bads"])
    # Extract adjacency matrix
    adjacency_matrix, ch_names = mne.channels.find_ch_adjacency(
        inst.info, ch_type
    )
    adjacency_matrix = adjacency_matrix.todense()
    adjacency_matrix = np.array(adjacency_matrix)
    # retrieve picks based on adjacency matrix
    picks = _picks_to_idx(inst.info, ch_names, exclude=[])
    # retrieve channel positions
    pos = inst._get_channel_positions(picks)
    interpolate_matrix = _make_interpolation_matrix(pos, pos)
    # retrieve data
    data = inst.get_data(picks=picks)
    if isinstance(inst, BaseEpochs):
        data = np.hstack(data)
    # Apply filter
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
            (inst._data.shape[0], len(picks), inst._data.shape[-1])
        )
        inst._data[:, picks, :] = data
    else:
        inst._data[picks] = data

    return inst


def _channel_spatial_filter(index, data, adjacency_vector, interpolate_matrix):
    neighbors_data = data[adjacency_vector == 1, :]
    neighbor_indices = np.argwhere(adjacency_vector == 1)
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
        if len(weights) == 0:
            # Not enough neighbors or all values are max/min
            continue
        # normalize weights
        weights = weights / np.linalg.norm(weights)
        # average
        channel_data[i] = np.average(data[keep_ind, i], weights=weights)
    return channel_data
