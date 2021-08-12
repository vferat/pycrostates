import numpy as np
from scipy.signal import find_peaks

import mne
from mne.io import BaseRaw
from mne.epochs import BaseEpochs
from mne.utils import _validate_type, logger, verbose, fill_doc, check_random_state
from mne.preprocessing.ica import _check_start_stop


def _extract_gfps(data, min_peak_distance=2):
    """Extract Gfp peaks from input data

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
    """
    if min_peak_distance < 1:
        raise(ValueError('min_peak_dist must be >= 1.'))
    gfp = np.std(data, axis=0)
    peaks, _ = find_peaks(gfp, distance=min_peak_distance)
    data_ =  data[:,peaks]
    return(data_)

@fill_doc
@verbose
def extract_gfp_peaks(inst, min_peak_distance=2, start=None, stop=None, reject_by_annotation=None, verbose=None):
    """Perform GFP peaks extraction

    Extract global field power peaks from :class:`mne.epochs.Epochs`or :class:`mne.io.Raw`.

    .. warning:: The temporal dimension of the output :class:`mne.io.Raw` object has been destroyed.
                 This object is a convinient container for gfp peaks and should not be used for standart MEEG analysis.

    Parameters
    ----------
    inst : :class:`mne.io.BaseRaw`, :class:`mne.Epochs`, :class:`mne.Evoked`
            Instance from which to extract gfp peaks.
    min_peak_dist : int
                    Required minimal horizontal distance (>= 1)
                    in samples between neighbouring peaks.
                    Smaller peaks are removed first until the
                    condition is fulfilled for all remaining peaks.
                    Default to 2.
    %(raw_tmin)s
    %(raw_tmax)s
    %(reject_by_annotation_raw)s
    %(verbose)s

    Returns
    -------
    raw : :class:`mne.io.BaseRaw`
        The Raw instance containing extracted gfp peaks
    """
    _validate_type(inst, (BaseRaw, BaseEpochs), 'inst', 'Raw or Epochs')
    if min_peak_distance < 1:
        raise(ValueError('min_peak_dist must be >= 1.'))
    if isinstance(inst, BaseRaw):
        reject_by_annotation = 'omit' if reject_by_annotation else None
        start, stop = _check_start_stop(inst, start, stop)
        data = inst.get_data(start=start, stop=stop,
                             reject_by_annotation=reject_by_annotation)
        peaks = _extract_gfps(data, min_peak_distance=min_peak_distance)

    if isinstance(inst, BaseEpochs):
        data = inst.get_data()
        peaks = list()
        for epoch in data:
            epoch_peaks = _extract_gfps(epoch, min_peak_distance=min_peak_distance)
            peaks.append(epoch_peaks)
        peaks = np.hstack(peaks)
    logger.info(f'{peaks.shape[1]} gfp peaks extracted out of {data.shape[1]} samples'
                f'({(peaks.shape[1] /data.shape[1] * 100):.2f}% of the original data)')
    raw_peaks = mne.io.RawArray(data=peaks, info=inst.info, verbose=False)
    return(raw_peaks)


@fill_doc
@verbose
def resample(inst, n_epochs=None, n_samples=None, coverage=None,
             replace=True, start=None, stop=None, reject_by_annotation=None,
             random_state=None, verbose=None):
    """Resample recording into epochs of random samples.

    Resample :class:`mne.io.Raw` or :class:`mne.epochs.Epochs`
    into n_epochs :class:`mne.io.Raw` each containing n_samples random samples of
    the originial recording.

    .. warning:: The temporal dimension of the output :class:`mne.io.Raw` objects has been destroyed.
                 These objects should not be used for standart MEEG analysis.

    Parameters
    ----------
    inst : :class:`mne.io.BaseRaw`, :class:`mne.Epochs`
        Instance from which to extract gfp peaks.
    n_epochs : int
        Number of epoch to draw.
    n_samples : int
        Length of each epoch (in samples).
    coverage: float (strictly positive)
        Ratio between resampling data size and size of the original recording.
        Can be > 1 if replace=True.
    replace: bool
        Whether or not to allow resampling with replacement.
        Default to True.
    %(raw_tmin)s
    %(raw_tmax)s
    %(reject_by_annotation_raw)s
    %(random_state)s
        As resampling can be non-deterministic it can be useful to fix the
        random state to have reproducible results.
    %(verbose)s

    Notes
    -----
    Only two of n_epochs, n_samples and coverage parameters must be defined, the non-defined one being computed during function execution.

    Returns
    -------
    raw : list of :class:`mne.io.BaseRaw`
        Raw objects each containing resampled data ( n_epochs raws of n_samples samples).
    """
    _validate_type(inst, (BaseRaw, BaseEpochs), 'inst', 'Raw or Epochs')
    random_state = check_random_state(random_state)

    if isinstance(inst, BaseRaw):
        reject_by_annotation = 'omit' if reject_by_annotation else None
        start, stop = _check_start_stop(inst, start, stop)
        data = inst.get_data(start=start, stop=stop,
                             reject_by_annotation=reject_by_annotation)

    if isinstance(inst, BaseEpochs):
        data = inst.get_data()
        data = np.hstack(data)

    n_times = data.shape[1]

    if len([x for x in [n_epochs, n_samples, coverage] if x is None]) >= 2:
        raise(ValueError('At least two of the [n_epochs, n_samples, coverage] must be defined'))

    if coverage is not None:
        if  coverage <= 0:
            raise(ValueError('Coverage must be strictly positive'))
    else:
        coverage = (n_epochs * n_samples) / n_times

    if n_epochs is None:
        n_epochs = int((n_times * coverage) / n_samples)

    if n_samples is None:
        n_samples = int((n_times * coverage) / n_epochs)

    if replace is False:
        if n_epochs * n_samples > n_times:
            raise(ValueError(f'''Can't draw {n_epochs} epochs of {n_samples} samples = {n_epochs * n_samples}'''
                             f'''samples without replacement: instance contains only {n_times} samples'''))

    logger.info(f'Resampling instance into {n_epochs} epochs of {n_samples} covering {coverage *100:2f}% of the data')

    if replace:
        indices = random_state.randint(0, n_samples, size=(n_epochs, n_samples))
    else:
        indices = np.arange(n_times)
        random_state.shuffle(indices)
        indices = indices[:n_epochs*n_samples]
        indices = indices.reshape((n_epochs, n_samples))

    data = data[:,indices]
    data = np.swapaxes(data,0,1)
    
    info = inst.info.copy()
    info['sfreq'] = -1
    resamples = list()
    for d in data:
        raw = mne.io.RawArray(d, info=info, verbose=False)
        resamples.append(raw)
    return(resamples)
 