"""Known image structure tensors: source-only UWB feasibility diagnostic."""
import numpy as np
from scipy.ndimage import sobel,gaussian_filter
from scipy.signal import hilbert


def image(raw):
    envelope=abs(hilbert(np.asarray(raw,dtype=np.float64),axis=-1))
    z=envelope.reshape(*envelope.shape[:-1],63,3).mean(-1)
    maximum=z.max(axis=(-2,-1),keepdims=True)
    return np.sqrt(np.divide(z,maximum,out=np.zeros_like(z),where=maximum>0))


def tensor(im):
    gt=sobel(im,axis=0,mode='reflect')/8;gr=sobel(im,axis=1,mode='reflect')/8
    return np.stack([gaussian_filter(z,sigma=2,truncate=4,mode='reflect') for z in [gt*gt,gt*gr,gr*gr]],-1)


def measures(z,threshold):
    a,b,d=z[...,0],z[...,1],z[...,2];trace=a+d;gap=np.sqrt(np.maximum((a-d)**2+4*b*b,0))
    coherence=np.divide(gap,trace,out=np.zeros_like(trace),where=trace>1e-12)
    oblique=np.divide(2*abs(b),gap,out=np.zeros_like(trace),where=gap>1e-12)
    eligible=(trace>=threshold)&(trace>1e-12)
    coherent=eligible&(coherence>=.6);usable=coherent&(oblique>=.3)
    denominator=int(eligible.sum())
    return dict(eligible=denominator,coherent=int(coherent.sum()),usable=int(usable.sum()),fraction=float(usable.sum()/max(denominator,1)),coverage=float(eligible.mean()))


def extract(raw,sid,seed=20261004):
    ims=image(raw);perm=np.random.default_rng(np.random.SeedSequence([seed,int(sid)])).permutation(90)
    time=np.arange(10,80,5);ranges=np.arange(10,53,5);out=[]
    for im in ims:
        pair=[]
        for z in [im,im[perm]]:pair.append(tensor(z)[time[:,None],ranges[None,:]].reshape(-1,3))
        out.append(pair)
    return np.asarray(out),perm


def summarize(tensors,users):
    rows=[]
    for user in range(3,9):
        ix=np.flatnonzero(users==user)
        for view in range(3):
            scores=[];counts=[]
            for n in ix:
                original=tensors[n,view,0];threshold=np.quantile(original[:,0]+original[:,2],.75)
                pair=[measures(tensors[n,view,arm],threshold) for arm in range(2)];scores.append([z['fraction'] for z in pair]);counts.append(pair)
            s=np.array(scores);mean=s.mean(0);passed=bool(mean[0]>=.2 and mean[0]-mean[1]>=.1)
            rows.append(dict(user=user,view=view,original=float(mean[0]),shuffled=float(mean[1]),gain=float(mean[0]-mean[1]),passed=passed,clips=len(ix),eligible=[sum(z[a]['eligible'] for z in counts) for a in range(2)],coverage=[float(np.mean([z[a]['coverage'] for z in counts])) for a in range(2)]))
    view_pass=[sum(z['passed'] for z in rows if z['view']==v)>=4 for v in range(3)]
    return dict(rows=rows,view_pass=view_pass,gate_passed=sum(view_pass)>=2)
