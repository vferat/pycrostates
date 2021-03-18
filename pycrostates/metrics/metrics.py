import itertools

import numpy as np
from scipy import stats
import mne
from mne.io import BaseRaw
from mne.epochs import BaseEpochs
from mne import Evoked
from mne.utils import _validate_type, verbose, warn, fill_doc
from mne.parallel import check_n_jobs, parallel_func

from ..utils import _corr_vectors
from ..clustering import BaseClustering

@verbose
def _compute_metrics(inst, modK,
                     norm_gfp,
                     reject_by_annotation,
                     half_window_size,
                     factor,
                     crit,
                     verbose=None):
    labels = modK.predict(inst, half_window_size=half_window_size,
                         factor=factor, crit=crit, verbose=verbose)
    
    if isinstance(inst, BaseRaw):
        data = inst.get_data()
    elif isinstance(inst, Evoked):
        data = inst.data
    gfp = np.std(data, axis=0)
    if norm_gfp:
        gfp = gfp / np.linalg.norm(gfp)

    segments = [(s, list(group)) for s,group in itertools.groupby(labels)]
    d = {}
    for s, state in enumerate(modK.cluster_centers):
        state_name = modK.names[s]
        arg_where = np.argwhere(labels == s+1)
        if len(arg_where) != 0:
            labeled_tp = data.T[arg_where][:,0,:].T
            labeled_gfp = gfp[arg_where][:,0]
            state_array = np.array([state]*len(arg_where)).transpose()
            corr = _corr_vectors(state_array, labeled_tp)
            gev = (labeled_gfp * corr) ** 2 / np.sum(gfp ** 2)
            s_segments = np.array([len(group) for s_, group in segments if s_ == s+1])
            occurence = len(s_segments) /len(data) *  inst.info['sfreq']
            timecov = np.sum(s_segments) / len(np.where(labels != 0)[0])
            durs = s_segments / inst.info['sfreq']

            d[f'{state_name}_dist_corr'] = corr
            d[f'{state_name}_mean_corr'] = np.mean(np.abs(corr)) 
            d[f'{state_name}_dist_gev'] = gev
            d[f'{state_name}_gev'] = np.sum(gev)
            d[f'{state_name}_timecov'] = timecov
            d[f'{state_name}_dist_durs'] = durs
            d[f'{state_name}_meandurs'] = np.mean(durs)
            d[f'{state_name}_occurences'] = occurence
        else:
            d[f'{state_name}_dist_corr'] = 0
            d[f'{state_name}_mean_corr'] = 0 
            d[f'{state_name}_dist_gev'] = 0
            d[f'{state_name}_gev'] = 0
            d[f'{state_name}_timecov'] = 0
            d[f'{state_name}_dist_durs'] = 0
            d[f'{state_name}_meandurs'] = 0
            d[f'{state_name}_occurences'] = 0
            
    d['unlabeled'] =  len(np.argwhere(labels == 0)) / len(gfp)
    d['segmentation'] = labels
    return(d)


@verbose
def compute_metrics(inst:mne.io.RawArray,
                    modK:BaseClustering,
                    norm_gfp:bool = True,
                    reject_by_annotation: bool = True,
                    half_window_size: int = 3, factor: int = 0,
                    crit: float = 10e-6,
                    n_jobs: int = 1,
                    verbose: str = None) -> dict:
    """Compute microstate metrics.
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
                   Mean number of segment assigned to a given state per second. This metrics is expressed in segment per second ( . / s)

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
    n_jobs = check_n_jobs(n_jobs)
    if isinstance(inst, list):
        if not all(isinstance(i, type(inst[0])) for i in inst):
            raise ValueError("All instances must be of the same type")
        if not all(i.info['sfreq'] == inst[0].info['sfreq'] for i in inst):
            raise ValueError("Not all instances have the same sampling frequency")
        inst = [i.pick(modK.picks) for i in inst]
        if n_jobs == 1:
            ds = [_compute_metrics(i, modK,
                            norm_gfp=reject_by_annotation,
                            reject_by_annotation=reject_by_annotation,
                            half_window_size=half_window_size,
                            factor=factor,
                            crit=crit,
                            verbose=verbose) for i in inst]
        else:
            parallel, p_fun, _ = parallel_func(_compute_metrics,
                                                    total=len(inst),
                                                    n_jobs=n_jobs)
            ds = parallel(p_fun(i, modK,
                                norm_gfp=reject_by_annotation,
                                reject_by_annotation=reject_by_annotation,
                                half_window_size=half_window_size,
                                factor=factor,
                                crit=crit,
                                verbose=verbose) for i in inst)
        return(ds)
    else:
        _validate_type(inst, (BaseRaw, BaseEpochs, Evoked), 'inst', 'Raw, Epochs or Evoked')
        d_ = _compute_metrics(inst, modK,
                              norm_gfp=reject_by_annotation,
                              reject_by_annotation=reject_by_annotation,
                              half_window_size=half_window_size,
                              factor=factor,
                              crit=crit,
                              verbose=verbose)
        return([d_])
    

