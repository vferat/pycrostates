"""This module contains functions to fetch remote datasets. All the dataset
fetchers are available in :mod:`pycrostates.datasets`. To download any of the
datasets, use the ``data_path`` function associated to this dataset.

All fetchers will check the default download location first to see if the
dataset is already on your computer, and only download it if necessary."""

from . import lemon

__all__: tuple[str, ...] = ("lemon",)
