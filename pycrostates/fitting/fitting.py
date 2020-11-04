import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal


class Segmentation():
    def __init__(self, states, labels, raw):
        self.states = states
        self.labels = labels
        self.raw = raw
        self.n_states = len(self.states)

    def plot(self, tmin=0.0, tmax=None):
        raw_ = self.raw.copy()
        raw_ = raw_.crop(tmin=tmin, tmax=tmax)
        gfp = np.std(raw_.get_data(), axis=0)
        times = raw_.times + tmin
        labels = self.labels[(times * raw_.info['sfreq']).astype(int)]
        n_states = self.n_states + 1
        cmap = plt.cm.get_cmap('plasma', n_states)

        fig = plt.figure(figsize=(10,4))
        plt.plot(times, gfp, color='black', linewidth=0.2)
        for state, color in zip(range(n_states), cmap.colors):
            w = np.where(labels[1:] == state)
            a = np.sort(np.append(w,  np.add(w, 1)))
            x = np.zeros(labels.shape)
            x[a] = 1
            x = x.astype(bool)
            plt.fill_between(times, gfp, color=color,
                             where=x, step=None, interpolate=False)
        norm = mpl.colors.Normalize(vmin=0, vmax=n_states)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm)
        plt.xlabel('Time (s)')
        plt.title('Segmentation')
        plt.autoscale(tight=True)
        
    def compute_metrics(self, norm_gfp=True):
        d = {}
        states_names = ['unlabeled'] + [f'state_{s+1}' for s in range(self.n_states)]
        segments = [(s, list(group)) for s,group in itertools.groupby(self.labels)][1:-1]
        for s in range(self.n_states + 1):
            state_name = states_names[s]
            s_segments = [len(group) for s_, group in segments if s_ == s]
            occurence = len(s_segments) /len(segments)
            timecov = np.sum(s_segments) / len(self.labels)
            meandur = np.mean(s_segments) / self.raw.info['sfreq']
            d[f'{state_name}_timecov'] =  timecov
            d[f'{state_name}_meandurs'] = meandur
            d[f'{state_name}_occurences'] = occurence
        return(d)
    


def segment(raw, states):
    data = raw.get_data()
    half_window_size = 3
    factor = 10
    crit = 10e-6
    S0 = 0  
    states = (states.T / np.linalg.norm(states, axis=1)).T
    data = (data.T / np.linalg.norm(data, axis=1)).T
    Ne, Nt = data.shape
    Nu = states.shape[0]
    Vvar = np.sum(data * data, axis=0)
    rmat = np.tile(np.arange(0,Nu), (Nt, 1)).T
    
    labels_all = np.argmax(np.abs(np.dot(states, data)), axis=0)
    
    w = np.zeros((Nu,Nt))
    w[(rmat == labels_all)] = 1
    e = np.sum(Vvar - np.sum(np.dot(w.T, states).T * data, axis=0) **2 / (Nt * (Ne - 1)))
    
    window = np.ones((1, 2*half_window_size+1))
    while True:
        Nb = scipy.signal.convolve2d(w, window, mode='same')
        x = (np.tile(Vvar,(Nu,1)) - (np.dot(states, data))**2) / (2* e * (Ne-1)) - factor * Nb   
        dlt = np.argmin(x, axis=0)
        
        labels_all = dlt
        w = np.zeros((Nu,Nt))
        w[(rmat == labels_all)] = 1
        Su = np.sum(Vvar - np.sum(np.dot(w.T, states).T * data, axis=0) **2) / (Nt * (Ne - 1))
        if np.abs(Su - S0) <= np.abs(crit * Su):
            break
        else:
            S0 = Su
    labels = labels_all +1
    seg = Segmentation(states, labels, raw)
    return(seg)