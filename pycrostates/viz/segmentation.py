from typing import Tuple, Union

import numpy as np
from mne import Epochs
from mne.io import BaseRaw
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

from ..utils._checks import _check_type, _check_axes


def plot_segmentation(
        labels: np.ndarray,
        inst: Union[BaseRaw],
        cluster_centers: np.ndarray,
        names: list = None,
        tmin: float = 0.0, tmax: float = None) -> Tuple[Figure, Axes]:
    _check_type(inst, (BaseRaw))
    inst = inst.copy()
    inst.crop(tmin=tmin, tmax=tmax)
    if isinstance(inst, BaseRaw):
        data = inst.get_data()
    elif isinstance(inst):
        data = inst.data  # TODO: I think it supports .get_data() nowadays
    gfp = np.std(data, axis=0)
    times = inst.times + tmin

    n_colors = 1 + len(cluster_centers)
    state_labels = [-1] + list(range(len(cluster_centers)))
    if not names:
        names = ['unlabeled'] + [str(k) for k in range(len(cluster_centers))]
    else:
        names = ['unlabeled'] + names

    labels = labels[(times * inst.info['sfreq']).astype(int)]
    cmap = plt.cm.get_cmap('plasma', n_colors)

    fig = plt.figure(figsize=(10, 4))
    ax = plt.plot(times, gfp, color='black', linewidth=0.2)
    for state, color in zip(state_labels, cmap.colors):
        w = np.where(labels[1:] == state)
        a = np.sort(np.append(w,  np.add(w, 1)))
        x = np.zeros(labels.shape)
        x[a] = 1
        x = x.astype(bool)
        plt.fill_between(times, gfp, color=color, where=x, step=None,
                         interpolate=False)
    norm = colors.Normalize(vmin=0, vmax=n_colors)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ticks=[i + 0.5 for i in range(n_colors)])
    cbar.ax.set(yticklabels=names)
    plt.xlabel('Time (s)')
    plt.title('Segmentation')
    plt.autoscale(tight=True)
    plt.show()
    return fig, ax


def plot_raw_segmentation(
        labels,
        inst,
        cluster_centers,
        cluster_names=None,
        tmin=0.0,
        tmax= None,
        cmap=None,
        ax=None,
        cbar_ax=None,
        block=False,
        **kwargs,
        ):
    """
    Plot segmentation

    Returns
    -------
    fig : Figure
        Matplotlib figure(s) on which topographic maps are plotted.
    """
    _check_type(labels, (np.ndarray, ), 'labels')
    _check_type(inst, (BaseRaw,), 'instance')
    _check_type(cluster_centers, (np.ndarray, ), 'cluster_centers')
    _check_type(cluster_names, (None, list, tuple), 'cluster_names')
    _check_type(ax, (None, Axes, tuple), 'ax')
    _check_type(cbar_ax, (None, Axes, tuple), 'cbar_ax')

    if ax is not None:
        _check_axes(ax)
    if cbar_ax is not None:
        _check_axes(cbar_ax)
    _check_type(block, (bool, ), 'block')

    # check cluster_names
    if cluster_names is None:
        cluster_names = [
            str(k) for k in range(1, cluster_centers.shape[0] + 1)]
    if len(cluster_names) != cluster_centers.shape[0]:
        raise ValueError(
            "Argument 'cluster_centers' and 'cluster_names' should have the "
            "same number of elements.")

    if ax is None:
        f, ax = plt.subplots(1, 1)
    else:
        f = ax.get_figure()

    # remove show from kwargs passed to topoplot
    show = True if 'show' not in kwargs else kwargs['show']
    _check_type(show, (bool, ), 'show')
    kwargs = {key: value for key, value in kwargs.items()
              if key not in ('show', )}

    inst = inst.copy()
    inst.crop(tmin=tmin, tmax=tmax)
    data = inst.get_data()
    gfp = np.std(data, axis=0)
    times = inst.times + tmin

    n_colors = 1 + len(cluster_centers)
    state_labels = [-1] + list(range(len(cluster_centers)))
    if not cluster_names:
        cluster_names = ['unlabeled'] + [str(k) for k in range(len(cluster_centers))]
    else:
        cluster_names = ['unlabeled'] + cluster_names

    labels = labels[(times * inst.info['sfreq']).astype(int)]
    cmap = plt.cm.get_cmap(cmap, n_colors)

    ax.plot(times, gfp, color='black', linewidth=0.2)
    for state, color in zip(state_labels, cmap.colors):
        w = np.where(labels[1:] == state)
        a = np.sort(np.append(w,  np.add(w, 1)))
        x = np.zeros(labels.shape)
        x[a] = 1
        x = x.astype(bool)
        ax.fill_between(times, gfp, color=color, where=x, step=None,
                         interpolate=False)
    ax.set_xlabel('Time (s)')
    ax.set_title('Segmentation')
    ax.autoscale(tight=True)
    # Color bar
    norm = colors.Normalize(vmin=0, vmax=n_colors)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    colorbar = plt.colorbar(sm, cax=cbar_ax, ax=ax, ticks=[i + 0.5 for i in range(n_colors)])
    colorbar.ax.set_yticklabels(cluster_names)

    if show:
        plt.show(block=block)
    return f


def plot_epoch_segmentation(
        labels,
        inst,
        cluster_centers,
        cluster_names=None,
        cmap=None,
        ax=None,
        cbar_ax=None,
        block=False,
        **kwargs,
        ):
    """
    Plot segmentation

    Returns
    -------
    fig : Figure
        Matplotlib figure(s) on which topographic maps are plotted.
    """
    _check_type(labels, (np.ndarray, ), 'labels')
    _check_type(inst, (Epochs,), 'instance')
    _check_type(cluster_centers, (np.ndarray, ), 'cluster_centers')
    _check_type(cluster_names, (None, list, tuple), 'cluster_names')
    _check_type(ax, (None, Axes, tuple), 'ax')
    _check_type(cbar_ax, (None, Axes, tuple), 'cbar_ax')

    if ax is not None:
        _check_axes(ax)
    if cbar_ax is not None:
        _check_axes(cbar_ax)
    _check_type(block, (bool, ), 'block')

    # check cluster_names
    if cluster_names is None:
        cluster_names = [
            str(k) for k in range(1, cluster_centers.shape[0] + 1)]
    if len(cluster_names) != cluster_centers.shape[0]:
        raise ValueError(
            "Argument 'cluster_centers' and 'cluster_names' should have the "
            "same number of elements.")

    if ax is None:
        f, ax = plt.subplots(1, 1)
    else:
        f = ax.get_figure()

    # remove show from kwargs passed to topoplot
    show = True if 'show' not in kwargs else kwargs['show']
    _check_type(show, (bool, ), 'show')
    kwargs = {key: value for key, value in kwargs.items()
              if key not in ('show', )}

    inst = inst.copy()
    data = inst.get_data()
    data_ = np.swapaxes(data, 0, 1)
    data_ = data_.reshape(data_.shape[0], -1)

    gfp = np.std(data_, axis=0)
    gfp = np.std(data_, axis=0)

    ts = np.arange(0, data_.shape[-1])
    
    x_ticks = np.linspace(0, data_.shape[-1], len(inst)+1)
    x_ticks += int(data.shape[-1] / 2)
    x_tick_labels = [str(i) for i in range(len(inst))] + ['']
    
    n_colors = 1 + len(cluster_centers)
    state_labels = [-1] + list(range(len(cluster_centers)))
    if not cluster_names:
        cluster_names = ['unlabeled'] + [str(k) for k in range(len(cluster_centers))]
    else:
        cluster_names = ['unlabeled'] + cluster_names

    labels_ = labels.reshape(-1)
    cmap = plt.cm.get_cmap(cmap, n_colors)

    ax.plot(ts, gfp, color='black', linewidth=0.2)
    for state, color in zip(state_labels, cmap.colors):
        w = np.where(labels_[1:] == state)
        a = np.sort(np.append(w,  np.add(w, 1)))
        x = np.zeros(labels_.shape)
        x[a] = 1
        x = x.astype(bool)
        ax.fill_between(ts, gfp, color=color, where=x, step=None,
                        interpolate=False)

    ax.set_xticks(x_ticks, x_tick_labels)
    ax.set_xlabel('Epochs')
    ax.set_title('Segmentation')
    # Epoch lines
    x = np.linspace(0, data_.shape[-1], data.shape[0]+1)
    ax.vlines(x, 0, gfp.max(), linestyles='dashed', colors='black')
    
    ax.autoscale(tight=True)
    # Color bar
    norm = colors.Normalize(vmin=0, vmax=n_colors)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    colorbar = plt.colorbar(sm, cax=cbar_ax, ax=ax, ticks=[i + 0.5 for i in range(n_colors)])
    colorbar.ax.set_yticklabels(cluster_names)

    if show:
        plt.show(block=block)
    return f
