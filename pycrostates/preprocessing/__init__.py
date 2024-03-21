"""This module contains functions to preprocess data for microstates analysis.
General preprocessing, such as filtering, interpolation, ICA, ... should be
applied prior to those functions using the MNE :mod:`~mne.preprocessing`
module."""

from .extract_gfp_peaks import extract_gfp_peaks
from .resample import resample
from .spatial_filter import apply_spatial_filter

__all__: tuple[str, ...] = ("apply_spatial_filter", "extract_gfp_peaks", "resample")
