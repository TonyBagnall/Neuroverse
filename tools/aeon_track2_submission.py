"""Single-file, NumPy-only inference for an aeon-trained multivariate MiniROCKET.

The MiniROCKET transform below is adapted from aeon1.5.0 _minirocket.py,
itself adapted from Angus Dempster's MiniROCKET implementation. Redistribution
must include the accompanying aeon BSD3-Clause licence (LICENSE.txt).
No aeon, numba, sklearn, pickle, network or runtime installation is needed.
"""
from itertools import combinations
from pathlib import Path
import numpy as np
from benchmark_utils.base_solver import CompetSolver


def transform(x,parameters):
    """Apply exported MiniROCKET parameters; retain original feature ordering."""
    x=np.asarray(x,dtype=np.float32)
    counts,channel_indices,dilations,features_per_dilation,biases=parameters
    indices=np.asarray(list(combinations(range(9),3)),dtype=np.int32)
    output=np.empty((len(x),len(biases)),dtype=np.float32)
    for i,case in enumerate(x):
        a=-case;g=3*case;feature_start=0;combination=0;channel_start=0
        channels,points=case.shape
        for j,dilation in enumerate(dilations):
            dilation=int(dilation);padding=4*dilation;n_features=int(features_per_dilation[j])
            alpha=a.copy();gamma=np.zeros((9,channels,points),dtype=np.float32);gamma[4]=g
            start=dilation;end=points-padding
            for index in range(4):
                alpha[:,-end:]=alpha[:,-end:]+a[:,:end]
                gamma[index,:,-end:]=g[:,:end];end+=dilation
            for index in range(5,9):
                alpha[:,:-start]=alpha[:,:-start]+a[:,start:]
                gamma[index,:,:-start]=g[:,start:];start+=dilation
            for k,(i0,i1,i2) in enumerate(indices):
                channel_end=channel_start+int(counts[combination]);pick=channel_indices[channel_start:channel_end]
                convolution=alpha[pick]+gamma[i0,pick]+gamma[i1,pick]+gamma[i2,pick]
                convolution=convolution.sum(axis=0)
                if (j%2+k)%2:convolution=convolution[padding:-padding]
                end_feature=feature_start+n_features
                output[i,feature_start:end_feature]=(convolution[:,None]>biases[None,feature_start:end_feature]).mean(axis=0)
                feature_start=end_feature;combination+=1;channel_start=channel_end
    return output


class PortableMiniRocket:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as data:
            self.parameters=tuple(data[k].copy() for k in ['counts','channel_indices','dilations','features_per_dilation','biases'])
            self.scale=data['scale'].copy();self.coef=data['coef'].copy();self.intercept=data['intercept'].copy();self.classes=data['classes'].copy()
            self.input_shape=tuple(int(i) for i in data['input_shape'])
            self.channel_names=tuple(data['channel_names'].tolist()) if 'channel_names' in data else ()
        if self.classes.tolist()!=[0,1,2]:raise ValueError('Expected Track2 class indices0/1/2')
        if len(self.parameters[-1])!=len(self.scale) or self.coef.shape!=(3,len(self.scale)):raise ValueError('Inconsistent fitted weights')

    def reset_state(self):
        """Stateless model: all normalisation was fitted on public training data."""

    def features(self,x):return transform(x,self.parameters)

    def decision_function(self,x):
        features=self.features(x)
        # sklearn StandardScaler preserves float32 input precision when dividing.
        features/=self.scale
        return features@self.coef.T+self.intercept

    def predict(self,x):
        if hasattr(x,'detach'):x=x.detach().cpu().numpy()
        x=np.asarray(x)
        if x.ndim!=3 or tuple(x.shape[1:])!=self.input_shape or not np.isfinite(x).all():raise ValueError('Invalid EEG tensor')
        return self.classes[np.argmax(self.decision_function(x),axis=1)].astype(np.int64)


class Solver(CompetSolver):
    name='Aeon-MiniROCKET-portable'
    requirements=[]

    def load_model(self,meta):
        model=PortableMiniRocket(Path(meta['submission_dir'])/'weights.npz')
        if (meta['n_chans'],meta['n_times'])!=model.input_shape or meta['n_classes']!=3:raise ValueError('Submission input contract mismatch')
        if not np.isclose(meta['sfreq'],120):raise ValueError('Expected120Hz microvolt competition input')
        if model.channel_names and tuple(meta.get('ch_names',[]))!=model.channel_names:raise ValueError('Channel names/order differ from training')
        return model
