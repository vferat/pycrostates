import pytest
import numpy as np
from pycrostates.clustering import mod_Kmeans


def test_mod_Kmeans():
    modK = mod_Kmeans(n_jobs=10)
    data = np.random.normal(size=(64, 1000))
    modK.fit(data)
    assert modK.cluster_centers is not None