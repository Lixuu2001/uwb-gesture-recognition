"""Fixed two-element group projection; no fitting or label access."""
import numpy as np
import torch
from circular_signal import get_batch

Q=np.array([1,0,2,3,6,7,4,5,9,8,10,11])
S=[2,1,0]


def arms(original,swapped):
    mapped=swapped[:,Q]
    return dict(ridge=original,equivariant=(original+mapped)/2,invariant=(original+swapped)/2,mapped_swap=mapped)


@torch.no_grad()
def extract(net,cache,ids):
    original=[];swapped=[]
    for i in range(0,len(ids),64):
        x=get_batch(cache,ids[i:i+64],'legacy');p=net.encoder(x.reshape(-1,4,90,63)).reshape(len(x),3,-1)
        original.append(net.head[:3](p.flatten(1)).numpy());swapped.append(net.head[:3](p[:,S].flatten(1)).numpy())
    return np.concatenate(original).astype(np.float64),np.concatenate(swapped).astype(np.float64)


def decision(scores,c):
    out={}
    for k,gain in c['gate']['minimum_mean_gain'].items():
        d=np.array(scores[c['primary']])-scores[k]
        out[k]=dict(mean_gain=float(d.mean()),wins=int((d>1e-12).sum()),minimum_gain=float(d.min()),passed=bool(d.mean()>=gain and (d>1e-12).sum()>=c['gate']['minimum_wins'] and d.min()>=-c['gate']['maximum_paired_decline']))
    return out
