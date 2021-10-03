import numpy as np

def _corr_vectors(A, B, axis=0):
    # based on https://github.com/wmvanvliet/mne_microstates/blob/master/microstates.py
    # written by Marijn van Vliet <w.m.vanvliet@gmail.com>
    """Compute pairwise correlation of multiple pairs of vectors.
    Fast way to compute correlation of multiple pairs of vectors without
    computing all pairs as would with corr(A,B). Borrowed from Oli at Stack
    overflow. Note the resulting coefficients vary slightly from the ones
    obtained from corr due differences in the order of the calculations.
    (Differences are of a magnitude of 1e-9 to 1e-17 depending of the tested
    data).
    Parameters
    ----------
    A : ndarray, shape (n, m)
        The first collection of vectors
    B : ndarray, shape (n, m)
        The second collection of vectors
    axis : int
        The axis that contains the elements of each vector. Defaults to 0.
    Returns
    -------
    corr : ndarray, shape (m,)
        For each pair of vectors, the correlation between them.
    """
    if A.shape != B.shape:
        raise ValueError('all input arrays must have the same shape')
    # If maps is null, divide will not trhow an error.
    np.seterr(divide='ignore', invalid='ignore')
    An = A - np.mean(A, axis=axis)
    Bn = B - np.mean(B, axis=axis)
    An /= np.linalg.norm(An, axis=axis)
    Bn /= np.linalg.norm(Bn, axis=axis)
    corr = np.sum(An * Bn, axis=axis)
    corr = np.nan_to_num(corr, posinf=0, neginf=0)
    np.seterr(divide='warn', invalid='warn')
    return corr

def _check_ch_names(inst1,inst2,inst1_name='inst', inst2_name='inst'):
    if inst1.info['ch_names'] != inst2.info['ch_names']:
        raise ValueError(f'Inconsistent Channel found between {inst1_name} and  {inst2_name}')
    return()

def _check_reject_by_annotation(reject_by_annotation):
    if reject_by_annotation is False:
        reject_by_annotation = None
    elif reject_by_annotation is True:
        reject_by_annotation = 'omit'
    else:
        raise ValueError('reject_by_annotation must be a boolean.')
    return(reject_by_annotation)