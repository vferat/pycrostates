
import numpy as np
from scipy.signal import find_peaks
import mne
from mne.utils.check import _validate_type

def extract_gfps(X, min_peak_dist=2, return_indices=False):
    """ Extract Gfp peaks from input data
    Parameters
    ----------
    min_peak_dist : Required minimal horizontal distance (>= 1)
                    in samples between neighbouring peaks.
                    Smaller peaks are removed first until the
                    condition is fulfilled for all remaining peaks.
                    Default to 2.
    X : array-like, shape [n_channels, n_samples]
                The data to extrat Gfp peaks, row by row. scipy.sparse matrices should be
                in CSR format to avoid an un-necessary copy.
    return_indices : bool, optional (default: False)
                Return peaks indices instead of peak values.
        """
    if not isinstance(return_indices, bool):
        raise TypeError(f'return_indices must be a Boolean, but class {type(return_indices)} was found')
    gfp = np.std(X, axis=0)
    peaks, _ = find_peaks(gfp, distance=min_peak_dist)
    if return_indices is True:
        return(peaks)
    elif return_indices is False:
        gfp_peaks = X[:, peaks]
        return(gfp_peaks)




class GfpExtractor():
    """Extract Gfp peaks from input data.
    Parameters
    ----------
    min_peak_dist : Required minimal horizontal distance (>= 1)
                    in samples between neighbouring peaks.
                    Smaller peaks are removed first until the
                    condition is fulfilled for all remaining peaks.
                    Default to 2.
    """

    def __init__(self, min_peak_dist=2):
        self.min_peak_dist = min_peak_dist

    def transform(self, raw, return_indices=False):
        """Extract Gfp peaks from input array.
        Parameters
        ----------
        raw : instance of Raw
            Raw measurements to be decomposed.
        return_indices : bool, optional (default: False)
            Return peaks indices instead of peak values.
        """
        data = extract_gfps(raw.get_data(), min_peak_dist=self.min_peak_dist)
        new_raw = mne.io.RawArray(data, raw.info)
        return (new_raw)
