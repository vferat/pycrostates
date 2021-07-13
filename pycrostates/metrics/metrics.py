import itertools

import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.utils import _safe_indexing, check_X_y
from sklearn.metrics.cluster._unsupervised import check_number_of_labels
from sklearn.preprocessing import LabelEncoder
from ..utils import _corr_vectors

def compute_metrics_data(segmentation, data, maps, maps_names, sfreq, norm_gfp=True):
    """Compute microstate metrics.

    Compute the following micorstates metrics:
    'dist_corr': Distribution of correlations
                 Correlation values of each time point assigned to a given state.
    'mean_corr': Mean correlation
                Mean correlation value of each time point assigned to a given state.
    'dist_gev': Distribution of global explained variances
                Global explained variance values of each time point assigned to a given state.
    'gev':  Global explained variance
            Total explained variance expressed by a given state. It is the sum of global explained
            variance values of each time point assigned to a given state.
    'timecov': Time coverage
               The proportion of time during which a given state is active. This metric is expressed in percentage (%%).
    'dist_durs': Distribution of durations.
                 Duration of each segments assigned to a given state. Each value is expressed in seconds (s).
    'meandurs': Mean duration
                Mean temporal duration segments assigned to a given state. This metric is expressed in seconds (s).
    'occurences' : Occurences
                Mean number of segment assigned to a given state per second. This metrics is expressed in segment per second ( . / s).

    Parameters
    ----------
    inst : :class:`mne.io.BaseRaw`, :class:`mne.Evoked`, list
        Instance or list of instances containing data to predict.
    modK : :class:`BaseClustering`
        Modified K-Means Clustering algorithm use to segment data
    norm_gfp : bool
        Either or not to normalize globalfield power.
    half_window_size: int
        Number of samples used for the half windows size while smoothing labels.
        Window size = 2 * half_window_size + 1
    factor: int
        Factor used for label smoothing. 0 means no smoothing.
        Defaults to 0.
    crit: float
        Converge criterion. Default to 10e-6.

    %(reject_by_annotation_raw)s
    %(verbose)s

    Returns
    ----------
    dict : list of dic
        Dictionaries containing microstate metrics.
    """
    gfp = np.std(data, axis=0)
    if norm_gfp:
        gfp = gfp / np.linalg.norm(gfp)

    gfp = np.std(data, axis=0)
    segments = [(s, list(group)) for s,group in itertools.groupby(segmentation)]
    d = {}
    for s, state in enumerate(maps):
        state_name = maps_names[s]
        arg_where = np.argwhere(segmentation == s+1)
        if len(arg_where) != 0:
            labeled_tp = data.T[arg_where][:,0,:].T
            labeled_gfp = gfp[arg_where][:,0]
            state_array = np.array([state]*len(arg_where)).transpose()

            corr = _corr_vectors(state_array, labeled_tp)
            d[f'{state_name}_dist_corr'] = corr
            d[f'{state_name}_mean_corr'] = np.mean(np.abs(corr))

            gev = (labeled_gfp * corr) ** 2 / np.sum(gfp ** 2)
            d[f'{state_name}_dist_gev'] = gev
            d[f'{state_name}_gev'] = np.sum(gev)
            
            s_segments = np.array([len(group) for s_, group in segments if s_ == s+1])
            occurence = len(s_segments) /len(segmentation) *  sfreq
            d[f'{state_name}_occurences'] = occurence
            
            timecov = np.sum(s_segments) / len(np.where(segmentation != 0)[0])
            d[f'{state_name}_timecov'] = timecov
            
            durs = s_segments / sfreq
            d[f'{state_name}_dist_durs'] = durs
            d[f'{state_name}_meandurs'] = np.mean(durs)
        else:
            d[f'{state_name}_dist_corr'] = 0
            d[f'{state_name}_mean_corr'] = 0
            d[f'{state_name}_dist_gev'] = 0
            d[f'{state_name}_gev'] = 0
            d[f'{state_name}_timecov'] = 0
            d[f'{state_name}_dist_durs'] = 0
            d[f'{state_name}_meandurs'] = 0
            d[f'{state_name}_occurences'] = 0
    d['unlabeled'] =  len(np.argwhere(segmentation == 0)) / len(gfp)
    return(d)


def distance_matrix(X, Y=None):
    distances = np.abs(1 / np.corrcoef(X, Y)) - 1
    distances = np.nan_to_num(distances, copy=False, nan=10e300, posinf=10e300, neginf=-10e300)
    return(distances)

def silhouette(modK): #lower the best
    modK._check_fit()
    data = modK.fitted_data_
    labels = modK.labels_
    keep = np.linalg.norm(data.T, axis=1) != 0
    data = data[:, keep]
    labels = labels[keep]
    distances = distance_matrix(data.T)
    silhouette = silhouette_score(distances, labels, metric='precomputed')
    return(silhouette)


def _davies_bouldin_score(X, labels):
    """Computes the Davies-Bouldin score.
    https://github.com/scikit-learn/scikit-learn/blob/2beed5584/sklearn/metrics/cluster/_unsupervised.py#L303
    """
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
        intra_dists[k] = np.average(distance_matrix(
            cluster_k, [centroid]))

    centroid_distances = distance_matrix(centroids)

    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0.0

    centroid_distances[centroid_distances == 0] = np.inf
    combined_intra_dists = intra_dists[:, None] + intra_dists
    scores = np.max(combined_intra_dists / centroid_distances, axis=1)
    return np.mean(scores)

def davies_bouldin(modK): # lower the best
    modK._check_fit()
    data = modK.fitted_data_
    labels = modK.labels_
    keep = np.linalg.norm(data.T, axis=1) != 0
    data = data[:, keep]
    labels = labels[keep]
    davies_bouldin_score = _davies_bouldin_score(data.T, labels)
    return(davies_bouldin_score)


def calinski_harabasz(modK): # lower the best
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

def _dunn_score(X, labels): #lower the best
    """ Dunn index - FAST (using sklearn pairwise euclidean_distance function)
    https://github.com/jqmviegas/jqm_cvi
    
    Parameters
    ----------
    X : np.array
        np.array([N, p]) of all points
    labels: np.array
        np.array([N]) labels of all points
    """
    distances = distance_matrix(X)
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

def dunn(modK):
    modK._check_fit()
    data = modK.fitted_data_
    labels = modK.labels_
    keep = np.linalg.norm(data.T, axis=1) != 0
    data = data[:, keep]
    labels = labels[keep]
    score = _dunn_score(data.T, labels)
    return(score)