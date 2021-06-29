import numpy as np
from scipy.signal import find_peaks

import mne
from mne.io import BaseRaw
from mne.epochs import BaseEpochs
from mne import Evoked
from mne.utils import _validate_type, logger, verbose, warn, fill_doc, check_random_state
from mne.preprocessing.ica import _check_start_stop

def _extract_gfps(data, min_peak_distance=2):
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

    """
    if not min_peak_distance >= 1:
        raise(ValueError('min_peak_dist must be >= 1.'))
    gfp = np.std(data, axis=0)
    peaks, _ = find_peaks(gfp, distance=min_peak_distance)
    data_ =  data[:,peaks]
    return(data_)

@verbose
def resample(inst, n_epochs:int=None, n_samples:int=None, coverage:float=None, replace:bool=True, start=None, stop=None, reject_by_annotation=None, random_state=None, verbose=None):
    _validate_type(inst, (BaseRaw, BaseEpochs, Evoked), 'inst', 'Raw, Epochs or Evoked')
    random_state = check_random_state(random_state)
    
    if isinstance(inst, BaseRaw):
        reject_by_annotation = 'omit' if reject_by_annotation else None
        start, stop = _check_start_stop(inst, start, stop)
        data = inst.get_data(start=start, stop=stop,
                             reject_by_annotation=reject_by_annotation)
            
    if isinstance(inst, BaseEpochs):
        data = inst.get_data()
        data = np.hstack(data)
        
    if isinstance(inst, Evoked):
        data = inst.data
        
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
            raise(ValueError(f'''Can't draw {n_epochs} epochs of {n_samples} samples = {n_epochs * n_samples} samples without replacement: instance contains only {n_times} samples'''))
        
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
    resamples = list()
    for d in data:
        raw = mne.io.RawArray(d, info=inst.info, verbose=False)
        resamples.append(raw)
    return(resamples)
 
        
    