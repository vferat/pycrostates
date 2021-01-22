import itertools

import mne
import numpy as np
from scipy import stats
from ..utils import _corr_vectors

def compute_metrics(labels:np.ndarray, states:np.ndarray,
                    raw:mne.io.RawArray, norm_gfp:bool = True,
                    state_as_index:bool=False) -> dict:
    """[summary]

    Args:
        labels (np.ndarray): [description]
        n_states (int): [description]
        raw (mne.io.RawArray): [description]

    Returns:
        dict: [description]
    """
    data = raw.get_data()
    gfp = np.std(data, axis=0)
    if norm_gfp:
        gfp = gfp / np.linalg.norm(gfp)

    n_states = states.shape[0]
    states_names = [f'state_{s+1}' for s in range(n_states)]

    segments = [(s, list(group)) for s,group in itertools.groupby(labels)]
    good_segments = [(s, list(group)) for s,group in itertools.groupby(labels) if s != 0]
    
    d_ = {}
    ds = list()
    for s, state in enumerate(states):
        d = {}
        state_name = states_names[s]
        d['state'] = state_name
        arg_where = np.argwhere(labels == s+1)
        labeled_tp = data.T[arg_where][:,0,:].T
        labeled_gfp = gfp[arg_where][:,0]
        state_array = np.array([state]*len(arg_where)).transpose()
        corr = _corr_vectors(state_array, labeled_tp)
        
        gev = (labeled_gfp * corr) ** 2 / np.sum(gfp ** 2)
        s_segments = np.array([len(group) for s_, group in segments if s_ == s+1])
        occurence = len(s_segments) /len(good_segments)
        timecov = np.sum(s_segments) / len(np.where(labels != 0)[0])
        durs = s_segments / raw.info['sfreq']
 
        d['dist_corr'] = corr
        d['mean_corr'] = np.mean(np.abs(corr))    
        d['dist_gev'] =  gev
        d['gev'] = np.sum(gev)
        d['timecov'] =  timecov
        d['dist_durs'] = durs
        d['meandurs'] = np.mean(durs)
        d['occurences'] = occurence
        ds.append(d)
        
        d_[f'{state_name}_dist_corr'] = corr
        d_[f'{state_name}_mean_corr'] = np.mean(np.abs(corr)) 
        d_[f'{state_name}_dist_gev'] = gev
        d_[f'{state_name}_gev'] = np.sum(gev)
        d_[f'{state_name}_timecov'] = timecov
        d_[f'{state_name}_dist_durs'] = durs
        d_[f'{state_name}_meandurs'] = np.mean(durs)
        d_[f'{state_name}_occurences'] = occurence

        
    d = {}
    s_segments = [len(group) for s_, group in segments if s_ == 0]
    timecov = np.sum(s_segments) / len(np.where(labels != 0)[0])
    d['state'] = 'unlabeled'
    d['timecov'] =  timecov
    d_['unlabeled_timecov'] = timecov
    ds.append(d)
    
    if state_as_index:
        return(ds)
    else:
        return(d_)

