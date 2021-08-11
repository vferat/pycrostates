import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.utils import _safe_indexing, check_X_y
from sklearn.metrics.cluster._unsupervised import check_number_of_labels
from sklearn.preprocessing import LabelEncoder


def _distance_matrix(X, Y=None):
    distances = np.abs(1 / np.corrcoef(X, Y)) - 1
    distances = np.nan_to_num(distances, copy=False, nan=10e300, posinf=10e300, neginf=-10e300)
    return(distances)

def silhouette(modK): #lower the better
    """Compute the mean Silhouette Coefficient of a fitted clustering algorithm.
    This function is a proxy function for :func:`sklearn.metrics.silhouette_score` that applies directly to a
    fitted :class:´pycrostate.clustering.BaseClustering´. It uses the absolute spatial correlation for distance
    computations.

    Parameters
    ----------
    BaseClustering : :class:`pycrostate.clustering.BaseClustering`
            Fitted clustering algorithm from which to compute score.

    Returns
    -------
    silhouette : float
        Mean Silhouette Coefficient.

    Notes
    -----
    For more details regarding the implementation, please refere to :func:`sklearn.metrics.silhouette_score`.
    This proxy function uses metric="precomputed" with the absolute spatial correlation for distance
    computations.

    References
    ----------
    .. [1] `Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
       Interpretation and Validation of Cluster Analysis". Computational
       and Applied Mathematics 20: 53-65.
       <https://doi.org/10.1016/0377-0427(87)90125-7>`_
    """
    modK._check_fit()
    data = modK.fitted_data_
    labels = modK.labels_
    keep = np.linalg.norm(data.T, axis=1) != 0
    data = data[:, keep]
    labels = labels[keep]
    distances = _distance_matrix(data.T)
    silhouette = silhouette_score(distances, labels, metric='precomputed')
    return(silhouette)


def _davies_bouldin_score(X, labels):
    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples, _ = X.shape
    n_labels = len(le.classes_)
    check_number_of_labels(n_labels, n_samples)

    intra_dists = np.zeros(n_labels)
    centroids = np.zeros((n_labels, len(X[0])), dtype=float)
    for k in range(n_labels):
        cluster_k = _safe_indexing(X, labels == k)
        centroid = cluster_k.mean(axis=0)
        centroids[k] = centroid
        intra_dists[k] = np.average(_distance_matrix(
            cluster_k, [centroid]))

    centroid_distances = _distance_matrix(centroids)

    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0.0

    centroid_distances[centroid_distances == 0] = np.inf
    combined_intra_dists = intra_dists[:, None] + intra_dists
    scores = np.max(combined_intra_dists / centroid_distances, axis=1)
    return np.mean(scores)

def davies_bouldin(modK): # lower the better
    """"Computes the Davies-Bouldin score.
    This function is a proxy function for :func:`sklearn.metrics.davies_bouldin_score` that applies directly to a
    fitted :class:´pycrostate.clustering.BaseClustering´. It uses the absolute spatial correlation for distance
    computations.

    Parameters
    ----------
    BaseClustering : :class:`pycrostate.clustering.BaseClustering`
            Fitted clustering algorithm from which to compute score.

    Returns
    -------
    score : float
        The resulting Davies-Bouldin score.
 
    Notes
    -----
    For more details regarding the implementation, please refere to :func:`sklearn.metrics.davies_bouldin_score`.
    This function was modified in order to use the absolute spatial correlation for distance
    computations instead of euclidean distance.
        
    References
    ----------
    .. [1] Davies, David L.; Bouldin, Donald W. (1979).
       `"A Cluster Separation Measure"
       <https://doi.org/10.1109/TPAMI.1979.4766909>`__.
       IEEE Transactions on Pattern Analysis and Machine Intelligence.
       PAMI-1 (2): 224-227
    """
    modK._check_fit()
    data = modK.fitted_data_
    labels = modK.labels_
    keep = np.linalg.norm(data.T, axis=1) != 0
    data = data[:, keep]
    labels = labels[keep]
    davies_bouldin_score = _davies_bouldin_score(data.T, labels)
    return(davies_bouldin_score)


def calinski_harabasz(modK): # lower the better
    """"Compute the Calinski and Harabasz score.
    This function is a proxy function for :func:`sklearn.metrics.calinski_harabasz_score` that applies directly to a
    fitted :class:´pycrostate.clustering.BaseClustering´.

    Parameters
    ----------
    BaseClustering : :class:´pycrostate.clustering.BaseClustering´
            Fitted clustering algorithm from which to compute score.

    Returns
    -------
    score : float
        The resulting Davies-Bouldin score.
        
    Notes
    -----
    For more details regarding the implementation, please refere to :func:`sklearn.metrics.calinski_harabasz_score`.

    References
    ----------
    .. [1] `T. Calinski and J. Harabasz, 1974. "A dendrite method for cluster
       analysis". Communications in Statistics
       <https://doi.org/10.1080/03610927408827101>`_
    """
    modK._check_fit()
    data = modK.fitted_data_
    labels = modK.labels_
    keep = np.linalg.norm(data.T, axis=1) != 0
    data = data[:, keep]
    labels = labels[keep]
    score = calinski_harabasz_score(data.T, labels)
    return(score)


def _delta_fast(ck, cl, distances):
    values = distances[np.where(ck)][:, np.where(cl)]
    values = values[np.nonzero(values)]
    return np.min(values)

def _big_delta_fast(ci, distances):
    values = distances[np.where(ci)][:, np.where(ci)]
    return np.max(values)

def _dunn_score(X, labels): #lower the better
    # based on https://github.com/jqmviegas/jqm_cvi
    """"Compute the Dunn index.

    Parameters
    ----------
    X : np.array
        np.array([N, p]) of all points
    labels: np.array
        np.array([N]) labels of all points
    """
    distances = _distance_matrix(X)
    ks = np.sort(np.unique(labels))
    
    deltas = np.ones([len(ks), len(ks)])*1000000
    big_deltas = np.zeros([len(ks), 1])
    
    l_range = list(range(0, len(ks)))
    
    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = _delta_fast((labels == ks[k]), (labels == ks[l]), distances)
        
        big_deltas[k] = _big_delta_fast((labels == ks[k]), distances)

    di = np.min(deltas)/np.max(big_deltas)
    return di

def dunn(modK): #lower the better
    """"Compute the Dunn index score.


    Parameters
    ----------
    BaseClustering : :class:´pycrostate.clustering.BaseClustering´
            Fitted clustering algorithm from which to compute score.

    Returns
    -------
    score : float
        The resulting Davies-Bouldin score.
        
    Notes
    -----
    This function uses the absolute spatial correlation for distance.
        
    References
    ----------
    .. [1] ` J. C. Dunn†, 1974. "Well-Separated Clusters and Optimal Fuzzy Partitions".
        Journal of Cybernetics
       <https://doi.org/10.1080/01969727408546059>`_
    """
    modK._check_fit()
    data = modK.fitted_data_
    labels = modK.labels_
    keep = np.linalg.norm(data.T, axis=1) != 0
    data = data[:, keep]
    labels = labels[keep]
    score = _dunn_score(data.T, labels)
    return(score)