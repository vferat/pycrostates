from typing import List, Optional, Union

import mne
import numpy as np
from mne import BaseEpochs
from mne.channels.interpolation import _make_interpolation_matrix
from mne.io import BaseRaw
from mne.io.pick import _picks_to_idx, _picks_by_type
from mne.parallel import parallel_func
from mne.utils.check import _check_preload

from ..utils._checks import _check_n_jobs, _check_type, _check_value
from ..utils._docs import fill_doc
from ..utils._logs import logger, verbose

@fill_doc
@verbose
def apply_spatial_filter(
    inst: Union[BaseRaw, BaseEpochs],
    ch_type: str = "eeg",
    exclude_bads: bool = True,
    n_jobs: int = 1,
    verbose = None
):
    """ Operates in place."""
    # Checks
    _check_type(inst, (BaseRaw, BaseEpochs))
    _check_value(ch_type, ("eeg"), item_name="ch_type")
    n_jobs = _check_n_jobs(n_jobs)
    # check preload for Raw
    _check_preload(inst, "Apply spatial filter")
    # remove bad channels
    # Extract adjacency matrix
    adjacency_matrix, ch_names = mne.channels.find_ch_adjacency(
        inst.info, ch_type
    )
    adjacency_matrix = adjacency_matrix.todense()
    adjacency_matrix = np.array(adjacency_matrix)
    if exclude_bads:
        for c, chan in enumerate(ch_names):
            if chan in inst.info["bads"]:
                adjacency_matrix[c, :] = 0 # do not change bads
                adjacency_matrix[:, c] = 0 # don't use bads to interpolate
                print(adjacency_matrix)
    # retrieve picks based on adjacency matrix
    picks = dict(_picks_by_type(inst.info, exclude=[]))[ch_type]
    assert ch_names == inst.ch_names
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
            (len(picks), inst._data.shape[0], inst._data.shape[-1])).swapaxes(0,1)
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
        return(data[index])
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
