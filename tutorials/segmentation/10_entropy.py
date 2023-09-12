"""
Microstate Segmentation
=======================

This tutorial introduces .
"""

#%%
# .. include:: ../../../../links.inc

#%%
# Entropy
# ------------
#
# We start by fitting a modified K-means
# (:class:`~pycrostates.cluster.ModKMeans`) with a sample dataset. For more
# details about fitting a clustering algorithm, please refer to
# :ref:`this tutorial <sphx_glr_generated_auto_tutorials_cluster_00_modK.py>`.
#
# .. note::
#
#     The lemon datasets used in this tutorial is composed of EEGLAB files. To
#     use the MNE reader :func:`mne.io.read_raw_eeglab`, the ``pymatreader``
#     optional dependency is required. Use the following installation method
#     appropriate for your environment:
#
#     - ``pip install pymatreader``
#     - ``conda install -c conda-forge pymatreader``
#
#     Note that an environment created via the `MNE installers`_ includes
#     ``pymatreader`` by default.

# sphinx_gallery_thumbnail_number = 2
from matplotlib import pyplot as plt
from mne.io import read_raw_eeglab

from pycrostates.cluster import ModKMeans
from pycrostates.datasets import lemon

raw_fname = lemon.data_path(subject_id="010017", condition="EC")
raw = read_raw_eeglab(raw_fname, preload=False)
raw.crop(0, 60)  # crop the dataset to speed up computation
raw.load_data()
raw.set_eeg_reference("average")  # Apply a common average reference

ModK = ModKMeans(n_clusters=5, random_state=42)
ModK.fit(raw, n_jobs=2)
ModK.reorder_clusters(order=[4, 1, 3, 2, 0])
ModK.rename_clusters(new_names=["A", "B", "C", "D", "F"])
ModK.plot()

#%%
# Once a set of cluster centers has been fitted, It can be used to predict the
# microstate segmentation with the method
# :meth:`pycrostates.cluster.ModKMeans.predict`. It returns either a
# `~pycrostates.segmentation.RawSegmentation` or an
# `~pycrostates.segmentation.EpochsSegmentation` depending on the object to
# segment. Below, the provided instance is a continuous `~mne.io.Raw` object:

segmentation = ModK.predict(
    raw,
    reject_by_annotation=True,
    factor=10,
    half_window_size=10,
    min_segment_length=5,
    reject_edges=True,
)

#%%
# Entropy
# TODO: explain the concept of entropy and its application to MS analysis
h = segmentation.entropy(ignore_self=False)

#%%
# We can also ignore state repetitions (i.e. self-transitions) by setting the ignore_self to True.
# This is useful when you don't want to take state duration into account.
h = segmentation.entropy(ignore_self=True)

#%%
# Excess entropy
# TODO: explain the concept of excess entropy and its application to MS analysis and parameters.
from pycrostates.segmentation import excess_entropy_rate
import matplotlib.pyplot as plt

a, b, residuals, lags, joint_entropies = excess_entropy_rate(segmentation, history_length=12, ignore_self=False)

plt.figure()
plt.plot(lags, joint_entropies, '-sk')
plt.plot(lags, a*lags+b, '-b')
plt.title("Entropy rate & excess entropy")
plt.show()

#%%
# auto_information_function
# TODO: explain the auto_information_function and parameters.
from pycrostates.segmentation import auto_information_function
import numpy as np

lags, ai = auto_information_function(segmentation, lags=np.arange(1, 20), ignore_self=False, n_jobs=2)

plt.figure()
plt.plot(lags, ai, '-sk')
plt.title("Auto information function")
plt.show()

#%%
# partial_auto_information_function
# TODO: explain the partial_auto_information_function and parameters.
from pycrostates.segmentation import partial_auto_information_function
import numpy as np

lags, pai = partial_auto_information_function(segmentation, lags=np.arange(1, 5), ignore_self=False, n_jobs=1)

plt.figure()
plt.plot(lags, pai, '-sk')
plt.title("Partial Auto information function")
plt.show()


#%%
# References
# ----------
# .. footbibliography::
