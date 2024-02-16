"""This module contains the clustering algorithms supported by ``pycrostates``.
Each clustering algorithm is defined as a class with specific initialization
parameters, a ``fit`` method and a ``predict`` method.

The fitting method ``fit`` requires a dataset as a :class:`~mne.io.Raw`,
:class:`~mne.Epochs` or :class:`~pycrostates.io.ChData` object. Fitting will
train the clustering algorithm and determine the microstates maps associated
with this dataset.

The fitted clustering algorithm can then be used to determine the segmentation
of the same or of another dataset (recorded using the same system, with the
same channels) using the ``predict`` method. This method will return either a
:class:`~pycrostates.segmentation.RawSegmentation` or a
:class:`~pycrostates.segmentation.EpochsSegmentation` depending on the dataset
to segment."""

from . import utils  # noqa: F401
from .aahc import AAHCluster  # noqa: F401
from .kmeans import ModKMeans  # noqa: F401

__all__: tuple[str, ...] = (
    "ModKMeans",
    "AAHCluster",
)
