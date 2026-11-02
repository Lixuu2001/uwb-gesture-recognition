"""Fixed local orientation histogram controls, known CV operations."""
import numpy as np
from orientation_screen import image,tensor


def original_tensors(raw):
    return np.stack([tensor(im)[10:80:5,10:53:5].reshape(126,3) for im in image(raw)])


def histogram(angle,weight,denominator):
    # 2 time cells x3 range cells,9 cyclic bins over undirected normal angle.
    theta=angle.reshape(14,9);w=weight.reshape(14,9);out=[]
    for ti in range(2):
        for ri in range(3):
            a=theta[ti*7:(ti+1)*7,ri*3:(ri+1)*3].reshape(-1)
            z=w[ti*7:(ti+1)*7,ri*3:(ri+1)*3].reshape(-1)
            pos=(a+np.pi/2)/np.pi*9;lo=np.floor(pos).astype(int);fraction=pos-lo
            h=np.bincount(lo%9,weights=z*(1-fraction),minlength=9)+np.bincount((lo+1)%9,weights=z*fraction,minlength=9)
            d=denominator.reshape(14,9)[ti*7:(ti+1)*7,ri*3:(ri+1)*3].sum()
            out.append(h/d if d>1e-12 else np.zeros(9))
    return np.array(out)


def describe(tensors,sid,seed=20261004):
    out={k:[] for k in ['all','coherent','sign_null','cell_null']}
    for view,z in enumerate(tensors):
        a,b,d=z.T;trace=a+d;gap=np.sqrt(np.maximum((a-d)**2+4*b*b,0));coh=np.divide(gap,trace,out=np.zeros_like(trace),where=trace>1e-12);ob=np.divide(2*abs(b),gap,out=np.zeros_like(trace),where=gap>1e-12)
        eligible=(trace>=np.quantile(trace,.75))&(trace>1e-12);usable=eligible&(coh>=.6)&(ob>=.3)
        angle=.5*np.arctan2(2*b,a-d);denominator=trace*eligible
        allhist=histogram(angle,trace*coh*eligible,denominator);hist=histogram(angle,trace*coh*usable,denominator)
        sign=np.random.default_rng(np.random.SeedSequence([seed,int(sid),view,0])).choice([-1,1],size=126)
        sh=histogram(.5*np.arctan2(2*b*sign,a-d),trace*coh*usable,denominator)
        perm=np.random.default_rng(np.random.SeedSequence([seed,int(sid),view,1])).permutation(6)
        for k,v in [('all',allhist),('coherent',hist),('sign_null',sh),('cell_null',hist[perm])]:out[k].append(v.reshape(-1))
    return {k:np.concatenate(v) for k,v in out.items()}
