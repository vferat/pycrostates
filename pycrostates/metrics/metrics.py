import itertools
import numpy as np
import mne

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
        
        labeled_tp = data.T[np.argwhere(labels == s+1)][:,0,:].T
        labeled_gfp = gfp[np.argwhere(labels == s+1)][:,0]
        stack = np.vstack([state, labeled_tp.T])
        corr = np.corrcoef(stack)[0][1:]
        abs_corr = np.abs(corr)
        gev = abs_corr *  labeled_gfp**2 / np.sum(gfp ** 2)
        d['dist_gev'] =  gev
        d['dist_corr'] = abs_corr
        d['gev'] = np.sum(gev)
        d['mean_corr'] = np.mean(abs_corr)
        d['ev'] = np.sum(abs_corr) / len(gfp)
        s_segments = np.array([len(group) for s_, group in segments if s_ == s+1])
        occurence = len(s_segments) /len(good_segments)
        timecov = np.sum(s_segments) / len(np.where(labels != 0)[0])
        durs = s_segments / raw.info['sfreq']
        d['timecov'] =  timecov
        d['meandurs'] = np.mean(durs)
        d['dist_durs'] = durs
        d['occurences'] = occurence
        ds.append(d)
        
        d_[f'{state_name}_dist_gev'] = gev
        d_[f'{state_name}_dist_corr'] = abs_corr
        d_[f'{state_name}_gev'] = np.sum(gev)
        d_[f'{state_name}_mean_corr'] = np.mean(abs_corr)
        d_[f'{state_name}_ev'] =np.sum(abs_corr) / len(gfp)
        d_[f'{state_name}_timecov'] = timecov
        d_[f'{state_name}_meandurs'] = np.mean(durs)
        d_[f'{state_name}_dist_durs'] = durs
        d_[f'{state_name}_occurences'] = occurence

        
    d = {}
    s_segments = [len(group) for s_, group in segments if s_ == 0]
    timecov = np.sum(s_segments) / len(np.where(labels != 0)[0])
    d['state'] = 'unlabeled'
    d['timecov'] =  timecov
    d_['unlabeled_timecoverage'] = timecov
    ds.append(d)
    
    if state_as_index:
        return(ds)
    else:
        return(d_)

