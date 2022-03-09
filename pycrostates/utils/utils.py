from copy import deepcopy

import mne
import numpy as np

from ._logs import logger


def _corr_vectors(A, B, axis=0):
    # based on:
    # https://github.com/wmvanvliet/mne_microstates/blob/master/microstates.py
    # written by Marijn van Vliet <w.m.vanvliet@gmail.com>
    """Compute pairwise correlation of multiple pairs of vectors.
    Fast way to compute correlation of multiple pairs of vectors without
    computing all pairs as would with corr(A,B). Borrowed from Oli at
    StackOverflow. Note the resulting coefficients vary slightly from the ones
    obtained from corr due to differences in the order of the calculations.
    (Differences are of a magnitude of 1e-9 to 1e-17 depending on the tested
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
    corr : ndarray, shape (m, )
        For each pair of vectors, the correlation between them.
    """
    if A.shape != B.shape:
        raise ValueError('All input arrays must have the same shape')
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


def _copy_info(inst, sfreq):
    ch_names = inst.info['ch_names']
    ch_types = [mne.channel_type(inst.info, idx)
                for idx in range(0, inst.info['nchan'])]
    new_info = mne.create_info(ch_names, sfreq=sfreq, ch_types=ch_types)
    if inst.get_montage():
        montage = inst.get_montage()
        new_info.set_montage(montage)
    return new_info


def _compare_infos(info1, info2):
    """
    Checks that both info have the same channels.
    """
    if not info1['ch_names'] == info2['ch_names']:
        raise ValueError(
            'Instance to segment into microstates sequence does not have '
            'the same channels as the instance used for fitting.')

    # Extract loc arrays
    loc1 = list()
    for ch in info1['chs']:
        loc1.append(deepcopy(ch['loc']))
    loc2 = list()
    for ch in info2['chs']:
        loc2.append(deepcopy(ch['loc']))

    # Compare loc
    assert len(loc1) == len(loc2)  # sanity-check
    for l1, l2 in zip(loc1, loc2):
        if not np.allclose(l1, l2, equal_nan=True):
            logger.warning(
                'Instance to segment into microstates sequence does not have '
                'the same channels montage as the instance used for fitting. ')
            break

    # Compare attributes in chs
    if not all(ch1['kind'] == ch2['kind']
               for ch1, ch2 in zip(info1['chs'], info2['chs'])):
        logger.warning(
            'Instance to segment into microstates sequence does not have '
            'the same channels kinds as the instance used for fitting. ')
    if not all(ch1['unit'] == ch2['unit']
               for ch1, ch2 in zip(info1['chs'], info2['chs'])):
        logger.warning(
            'Instance to segment into microstates sequence does not have '
            'the same channels units as the instance used for fitting. ')
    if not all(ch1['coord_frame'] == ch2['coord_frame']
               for ch1, ch2 in zip(info1['chs'], info2['chs'])):
        logger.warning(
            'Instance to segment into microstates sequence does not have '
            'the same coordinate frames as the instance used for fitting. ')
