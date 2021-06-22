from mne.io.base import TimeMixin
from pycrostates import segmentation
from mne.utils import _validate_type, logger, verbose, warn, fill_doc, check_random_state
from mne.io import BaseRaw
from mne.epochs import BaseEpochs
from mne import Evoked

from ..metrics import compute_metrics_data
from ..viz import plot_cluster_centers, plot_segmentation

class BaseSegmentation():
    def __init__(self, segmentation, inst, cluster_centers, names=None):
        self.segmentation = segmentation
        self.inst = inst        
        self.cluster_centers = cluster_centers

        if names:
            if len(self.cluster_centers) == len(names):
                self.names = names
            else:
                raise ValueError('Clsuter_centers and cluster_centers_names must have the same length')
        else:
            self.names = [f'{c+1}' for c in range(len(cluster_centers))]
        
    def plot_cluster_centers(self):
        fig, axs = plot_cluster_centers(self.cluster_centers, self.inst.info, self.names)
        return(fig, axs)
    
    
class RawSegmentation(BaseSegmentation):
    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        _validate_type(self.inst, (BaseRaw), 'inst', 'Raw')
        
        data = self.inst.get_data()
        if data.shape[1] != len(self.segmentation):
            raise ValueError('Instance and segmentation must have the same number of samples.')

    def plot(self, tmin: float = 0.0, tmax: float = None):
        fig, ax = plot_segmentation(segmentation=self.segmentation,
                                    inst=self.inst,
                                    cluster_centers=self.cluster_centers,
                                    names=self.names,
                                    tmin=tmin,
                                    tmax=tmax)
        return(fig, ax)
    
    def compute_metrics(self, norm_gfp=True):
        d = compute_metrics_data(self.segmentation,
                                self.inst.get_data(),
                                self.cluster_centers,
                                self.names,
                                self.inst.info['sfreq'],
                                norm_gfp=norm_gfp)
        return(d)

class EpochsSegmentation(BaseSegmentation):
    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        _validate_type(self.inst, (BaseEpochs), 'inst', 'Epochs')
        
        data = self.inst.get_data()
        if data.shape[0] != self.segmentation.shape[0]:
            raise ValueError('Epochs and segmentation must have the number of epochs.')
        if data.shape[2] != self.segmentation.shape[1]:
            raise ValueError('Epochs and segmentation must have the number of samples per epoch.')

    def compute_metrics(self, norm_gfp=True):
        data = self.inst.get_data()
        data = data.reshape(data.shape[0], -1)
        d = compute_metrics_data(self.segmentation,
                                data,
                                self.cluster_centers,
                                self.maps_names,
                                self.inst.info['sfreq'],
                                norm_gfp=norm_gfp)
        return(d)
    
class EvokedSegmentation(BaseSegmentation):
    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        _validate_type(self.inst, (Evoked), 'inst', 'Evoked')
        
        data = self.inst.data
        if data.shape[1] != len(self.segmentation):
            raise ValueError('Instance and segmentation must have the same number of samples.')

    def plot(self, tmin: float = 0.0, tmax: float = None):
        fig, ax = plot_segmentation(segmentation=self.segmentation,
                                    inst=self.inst,
                                    names=self.names,
                                    tmin=tmin,
                                    tmax=tmax)
        return(fig, ax)
    
    def compute_metrics(self, norm_gfp=True):
        d = compute_metrics_data(self.segmentation,
                                self.data(),
                                self.cluster_centers,
                                self.maps_names,
                                self.inst.info['sfreq'],
                                norm_gfp=norm_gfp)
        return(d)