import itertools
import numpy as np
import mne
from typing import Tuple, Union


def compute_gev_and_corr(data, state):
    gfp = np.std(data, axis=0)
    norm_gfp = gfp / np.linalg.norm(gfp)
    stack = np.vstack([state, data.T])
    corr = np.corrcoef(stack)[0][1:]
    abs_corr = np.abs(corr)
    gev = abs_corr *  norm_gfp
    return gev, abs_corr
    
    
def compute_metrics(labels:np.ndarray, states:np.ndarray, raw:mne.io.RawArray, state_as_index=True) -> dict:
    """[summary]

    Args:
        labels (np.ndarray): [description]
        n_states (int): [description]
        raw (mne.io.RawArray): [description]

    Returns:
        dict: [description]
    """
    data = raw.get_data()
    n_states = states.shape[0]
    states_names = [f'state_{s+1}' for s in range(n_states)]
    
    segments = [(s, list(group)) for s,group in itertools.groupby(labels)]
    if state_as_index is False:
        d = {}
        for s, state in enumerate(states):
            state_name = states_names[s]
            
            labeled_tp = data.T[np.argwhere(labels == s+1)][:,0,:].T
            gev, corr = compute_gev_and_corr(labeled_tp, state)
            d[f'{state_name}_dist_gev'] =  gev
            d[f'{state_name}_dist_corr'] = corr
            d[f'{state_name}_gev'] = np.sum(gev)
            d[f'{state_name}_mean_corr'] = np.mean(corr)            
       
            s_segments = [len(group) for s_, group in segments if s_ == s+1]
            occurence = len(s_segments) /len(segments)
            timecov = np.sum(s_segments) / len(labels)
            durs = np.array(s_segments) / raw.info['sfreq']
            
            d[f'{state_name}_timecov'] =  timecov
            d[f'{state_name}_meandurs'] = np.mean(durs)
            d[f'{state_name}_dist_meandurs'] = durs
            d[f'{state_name}_occurences'] = occurence
            d[f'{state_name}_durations'] = s_segments

        s_segments = [len(group) for s_, group in segments if s_ == 0]
        occurence = len(s_segments) /len(segments)
        timecov = np.sum(s_segments) / len(labels)
        meandur = np.mean(s_segments) / raw.info['sfreq']
        d['unlabeled_timecov'] =  timecov
        d['unlabeled_meandurs'] = meandur
        d['unlabeled_occurences'] = occurence
        d['unlabeled_durations'] = s_segments
        return(d)
    
    else:
        ds = list()
        for s, state in enumerate(states):
            d = {}
            state_name = states_names[s]
            d['state'] = state_name
            
            labeled_tp = data.T[np.argwhere(labels == s+1)][:,0,:].T
            gev, corr = compute_gev_and_corr(labeled_tp, state)
            d['dist_gev'] =  gev
            d['dist_corr'] = corr
            d['gev'] = np.sum(gev)
            d['mean_corr'] = np.mean(corr)            

            s_segments = [len(group) for s_, group in segments if s_ == s+1]
            occurence = len(s_segments) /len(segments)
            timecov = np.sum(s_segments) / len(labels)
            durs = np.array(s_segments) / raw.info['sfreq']
            d['timecov'] =  timecov
            d['meandurs'] = np.mean(durs)
            d['dist_meandurs'] = durs
            d['occurences'] = occurence
            d['durations'] = s_segments
            ds.append(d)
            
        d = {}
        s_segments = [len(group) for s_, group in segments if s_ == 0]
        occurence = len(s_segments) /len(segments)
        timecov = np.sum(s_segments) / len(labels)
        meandur = np.mean(s_segments) / raw.info['sfreq']
        d['state'] = 'unlabeled'
        d['timecov'] =  timecov
        d['meandurs'] = meandur
        d['occurences'] = occurence
        d['durations'] = s_segments
        ds.append(d)
        return(ds) 
    
if __name__ == "__main__":
    from mne.datasets import sample
    from pycrostates.clustering import mod_Kmeans
    import mne
    data_path = sample.data_path()
    raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
    event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
    # Setup for reading the raw data
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    raw = raw.pick('eeg')
    raw = raw.filter(0, 40)
    raw = raw.crop(0, 60)
    raw.info['bads'].append('Fp1')
    modK = mod_Kmeans()
    modK.fit(raw, gfp=True, n_jobs=5, verbose=False)
    seg = modK.predict(raw, reject_by_annotation=True)
    d = compute_metrics(seg, modK.cluster_centers, raw)
    print(d)