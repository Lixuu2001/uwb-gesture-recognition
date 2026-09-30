"""Auditable closed-set experiment utilities. Does not import/change main.py."""
import sys, json, hashlib, time, argparse
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
RESULTS=BASE/'results_clean'
sys.path.insert(0,str(BASE/'vendor'))
from uwb_osr.data import _analytic, doppler_map

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def dump(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(path)

def quality_mask(y,s):
    # Source-level contradictions verified in both original MAT and CSV.
    # Exclude BOTH conflicting labels; never guess which of G11/G12 is true.
    return ~(((s==2)&np.isin(y,[10,11])) | ((s==6)&(y==3)))

def split(y,s,held,seed):
    """Seven training subjects, within-subject validation; held person untouched."""
    rng=np.random.default_rng(seed)
    tr=[]; va=[];good=quality_mask(y,s)
    for subj in sorted(set(s)-{held}):
        for lab in sorted(set(y)):
            ids=np.where((s==subj)&(y==lab)&good)[0]; rng.shuffle(ids)
            if not len(ids):continue
            n=max(1,int(round(len(ids)*.1)))
            va.extend(ids[:n]); tr.extend(ids[n:])
    return np.sort(tr),np.sort(va),np.where((s==held)&good)[0]

def norm(x,absolute=False):
    x=np.abs(x) if absolute else x
    return (x-x.mean(axis=(-2,-1),keepdims=True))/(x.std(axis=(-2,-1),keepdims=True)+1e-6)

def transform(x,mode):
    """Input N,V,T,M. Exact old transforms before float16 storage."""
    x=np.asarray(x,dtype=np.float32)
    if mode=='signed': return norm(x)
    amp=norm(x,True)
    if mode=='off': return amp
    if mode=='duplicate': return np.concatenate([amp,amp],-1)
    vel=norm(np.stack([doppler_map(a) for a in x]))
    if mode=='only': return vel
    if mode=='concat': return np.concatenate([amp,vel],-1)
    raise ValueError(mode)

def phase_features(x):
    """Amplitude-weighted wrapped phase increments (radians/frame), NOT m/s.

    Analytic construction assumes phase stability; extended scatterers and alias
    are not resolved. Full-clip static removal is suitable only for cut clips.
    """
    z=_analytic(x.astype(np.float64)); z-=z.mean(axis=1,keepdims=True)
    d=z[:,1:]*z[:,:-1].conj(); w=np.sqrt(np.abs(d))
    den=w.sum(-1)+1e-12
    c=(np.angle(d)*w).sum(-1)/den
    q=np.abs((d/(np.abs(d)+1e-12)*w).sum(-1))/den
    return (np.concatenate([c[:,:1],c],1).astype('float32'),
            np.concatenate([q[:,:1],q],1).astype('float32'))

def motion_stats(c):
    values=[c.mean(-1),c.std(-1),*np.quantile(c,[.1,.5,.9],axis=-1),
            (c>0).mean(-1),*[b.mean(-1) for b in np.array_split(c,6,axis=-1)],
            np.abs(c).mean(-1)]
    return np.stack(values,-1).astype('float32')

def prepare():
    from numpy.lib.format import open_memmap
    out=BASE/'prepared'; out.mkdir(exist_ok=True)
    manifest={'protocol':'12-class, cache order verified separately','hashes':{},'counts':{}}
    for dtype in ('clutterremoved','raw'):
        xp=ROOT/f'cache/X_{dtype}.npy'; x=np.load(xp,mmap_mode='r')
        y=np.load(ROOT/f'cache/y_{dtype}.npy'); s=np.load(ROOT/f'cache/subj_{dtype}.npy')
        manifest['counts'][dtype]={f'{p}:{g}':int(((s==p)&(y==g)).sum()) for p in np.unique(s) for g in np.unique(y)}
        manifest['hashes'][str(xp)]=digest(xp)
        for key in ('y','subj'):
            p=ROOT/f'cache/{key}_{dtype}.npy'; manifest['hashes'][str(p)]=digest(p)
        finite=True; duplicates={}; seen={}
        for i in range(len(x)):
            row=np.asarray(x[i]); finite=finite and bool(np.isfinite(row).all())
            h=hashlib.blake2b(row.tobytes(),digest_size=16).hexdigest()
            if h in seen: duplicates[str(i)]=seen[h]
            seen[h]=i
        manifest[dtype]={'shape':list(x.shape),'finite':finite,'exact_duplicate_units':duplicates}
        if dtype=='raw': continue
        np.save(out/'y.npy',y);np.save(out/'subj.npy',s)
        maps={m:open_memmap(out/f'{m}.npy',mode='w+',dtype='float16',shape=x.shape) for m in ('off','only','signed')}
        c=open_memmap(out/'motion.npy',mode='w+',dtype='float32',shape=(len(x),3,90))
        q=open_memmap(out/'quality.npy',mode='w+',dtype='float32',shape=(len(x),3,90))
        st=open_memmap(out/'stats.npy',mode='w+',dtype='float32',shape=(len(x),3,13))
        amp=open_memmap(out/'amp_stats.npy',mode='w+',dtype='float32',shape=(len(x),3,210))
        for start in range(0,len(x),32):
            b=np.asarray(x[start:start+32]); end=start+len(b)
            a=transform(b,'off'); maps['off'][start:end]=a
            maps['only'][start:end]=transform(b,'only');maps['signed'][start:end]=transform(b,'signed')
            # 10 slow-time x 21 range cells, preserves coarse shape; train-fold scaler later.
            amp[start:end]=a.reshape(len(b),3,10,9,21,9).mean((3,5)).reshape(len(b),3,210)
            for j,row in enumerate(b):
                cc,qq=phase_features(row);c[start+j]=cc;q[start+j]=qq;st[start+j]=motion_stats(cc)
            if start%1024==0: print('prepare',start,len(x),flush=True)
        for arr in [*maps.values(),c,q,st,amp]: arr.flush()
        del maps,c,q,st,amp
    for p in (BASE/'vendor').rglob('*.py'):manifest['hashes'][str(p.relative_to(BASE))]=digest(p)
    dump(BASE/'results/data_manifest.json',manifest)
    print('PREPARE_COMPLETE',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare']);a=ap.parse_args()
    prepare()
