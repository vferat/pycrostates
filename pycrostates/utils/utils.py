"""Utils functions."""

from copy import deepcopy

import numpy as np

from ._logs import logger


def _correlation(A, B, ignore_polarity=True):
    """Compute pairwise correlation of multiple pairs of vectors."""
    # correlation with a given vector
    if A.ndim == 1 and B.ndim == 1:
        A = np.tile(A.T, (1, 1)).T
        B = np.tile(B.T, (1, 1)).T
        print(A.shape)
    else:
        if B.ndim == 1:
            B = np.tile(B.T, (A.shape[1], 1)).T
        if A.ndim == 1:
            A = np.tile(A.T, (B.shape[1], 1)).T
    # correlation element-wise (A1,B1), (A2,B2), ...
    if A.shape == B.shape:
        # If maps is null, divide will not throw an error.
        np.seterr(divide="ignore", invalid="ignore")
        An = A - np.mean(A, axis=0)
        Bn = B - np.mean(B, axis=0)
        An /= np.linalg.norm(An, axis=0)
        Bn /= np.linalg.norm(Bn, axis=0)
        corr = np.sum(An * Bn, axis=0)
        corr = np.nan_to_num(corr, posinf=0, neginf=0)
        np.seterr(divide="warn", invalid="warn")
    else:
        raise ValueError("All input arrays must have the same shape")

    if ignore_polarity:
        corr = np.abs(corr)
    return corr


def _distance(X, Y, ignore_polarity=True):
    """Compute pairwise distance of multiple pairs of vectors."""
    corr = _correlation(X, Y, ignore_polarity=True)
    dist = 1 - corr
    return dist


def _gev(data, maps, segmentation):
    map_corr = _correlation(data, maps[segmentation].T)
    gev = np.sum((data * map_corr) ** 2) / np.sum(data**2)
    return gev


def _compare_infos(cluster_info, inst_info):
    """Check that channels in cluster_info are all present in inst_info."""
    for ch in cluster_info["ch_names"]:
        if ch not in inst_info["ch_names"]:
            raise ValueError(
                "Instance to segment into microstates sequence does not have "
                "the same channels as the instance used for fitting."
            )

    # Extract loc arrays
    cluster_loc = list()
    for ch in cluster_info["chs"]:
        cluster_loc.append((ch["ch_name"], deepcopy(ch["loc"])))
    inst_loc = list()
    for ch in inst_info["chs"]:
        if ch["ch_name"] in cluster_info["ch_names"]:
            inst_loc.append((ch["ch_name"], deepcopy(ch["loc"])))
    cluster_loc = [loc[1] for loc in sorted(cluster_loc, key=lambda x: x[0])]
    inst_loc = [loc[1] for loc in sorted(inst_loc, key=lambda x: x[0])]

    # Compare loc
    assert len(cluster_loc) == len(inst_loc)  # sanity-check
    for l1, l2 in zip(cluster_loc, inst_loc):
        if not np.allclose(l1, l2, equal_nan=True):
            logger.warning(
                "Instance to segment into microstates sequence does not have "
                "the same channels montage as the instance used for fitting. "
            )
            break

    # Compare attributes in chs
    cluster_kinds = []
    cluster_units = []
    cluster_coord_frame = []
    for ch in cluster_info["chs"]:
        cluster_kinds.append((ch["ch_name"], ch["kind"]))
        cluster_units.append((ch["ch_name"], ch["unit"]))
        cluster_coord_frame.append((ch["ch_name"], ch["coord_frame"]))

    inst_kinds = []
    inst_units = []
    inst_coord_frames = []
    for ch in inst_info["chs"]:
        if ch["ch_name"] in cluster_info["ch_names"]:
            inst_kinds.append((ch["ch_name"], ch["kind"]))
            inst_units.append((ch["ch_name"], ch["unit"]))
            inst_coord_frames.append((ch["ch_name"], ch["coord_frame"]))

    cluster_kinds = [elt[1] for elt in sorted(cluster_kinds, key=lambda x: x[0])]
    cluster_units = [elt[1] for elt in sorted(cluster_units, key=lambda x: x[0])]
    cluster_coord_frame = [
        elt[1] for elt in sorted(cluster_coord_frame, key=lambda x: x[0])
    ]
    inst_kinds = [elt[1] for elt in sorted(inst_kinds, key=lambda x: x[0])]
    inst_units = [elt[1] for elt in sorted(inst_units, key=lambda x: x[0])]
    inst_coord_frames = [
        elt[1] for elt in sorted(inst_coord_frames, key=lambda x: x[0])
    ]

    if not all(kind1 == kind2 for kind1, kind2 in zip(cluster_kinds, inst_kinds)):
        logger.warning(
            "Instance to segment into microstates sequence does not have "
            "the same channels kinds as the instance used for fitting. "
        )
    if not all(unit1 == unit2 for unit1, unit2 in zip(cluster_units, inst_units)):
        logger.warning(
            "Instance to segment into microstates sequence does not have "
            "the same channels units as the instance used for fitting. "
        )
    if not all(f1 == f2 for f1, f2 in zip(cluster_coord_frame, inst_coord_frames)):
        logger.warning(
            "Instance to segment into microstates sequence does not have "
            "the same coordinate frames as the instance used for fitting. "
        )
