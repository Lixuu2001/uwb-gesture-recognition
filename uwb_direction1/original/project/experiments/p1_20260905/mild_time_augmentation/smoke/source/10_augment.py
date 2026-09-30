"""Source-only augmentation with signed CIR warping and fresh phase features."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent; BASE=HERE.parent
_spec=importlib.util.spec_from_file_location('_mild_time_pipeline',BASE/'pipeline.py')
_pipeline=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_pipeline)
legacy_transform=_pipeline.transform

def raw_warp(x,amount):
    x=np.asarray(x,dtype=np.float32)
    if x.ndim!=4 or x.shape[1:]!=(1,90,189): raise ValueError('Require Right CIR [N,1,90,189]')
    t=np.arange(90,dtype=np.float64); g=t+amount*np.sin(2*np.pi*t/89)
    g[0]=0; g[-1]=89
    if not np.all(np.diff(g)>0): raise ValueError('Nonmonotone warp')
    lo=np.floor(g).astype(int); hi=np.minimum(lo+1,89)
    f=(g-lo).astype(np.float32)[None,None,:,None]
    # Advanced indexing can change memory order and NumPy reduction order.
    # Restore the original contiguous T,M layout before legacy normalization.
    return np.ascontiguousarray(x[:,:,lo]*(1-f)+x[:,:,hi]*f)

def prepare_input(x,amount):
    return legacy_transform(raw_warp(x,amount),'concat').astype(np.float16)

def epoch_schedule(train_ids,seed):
    rng=np.random.default_rng(seed)
    selected=rng.random(len(train_ids))<.5
    direction=rng.choice(np.array([-1,1],dtype=np.int8),size=len(train_ids))
    return np.where(selected,direction,0).astype(np.int8)

def sampler_seed(sampler_state,fold):
    value=json.dumps({'sampler':sampler_state['rng'],'fold':fold,'augmentation_seed':20260908},sort_keys=True)
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8],'little')

class AugmentReader:
    def __init__(self,base,cache,enabled):
        self.base=base; self.cache=cache; self.enabled=enabled
        self.partitions=base.partitions
        self.plan=np.zeros(len(base.amplitude),dtype=np.int8)
        self.trace=hashlib.sha256(); self.counts={-1:0,0:0,1:0}
    def configure(self,seed):
        self.plan.fill(0)
        ids=self.partitions['train']; self.plan[ids]=epoch_schedule(ids,seed)
        self.trace=hashlib.sha256(); self.counts={-1:0,0:0,1:0}
    def load_ids(self,ids,role,mode,device='cuda'):
        # Original guard executes before any augmentation cache lookup.
        x=self.base.load_ids(ids,role,mode,device)
        if role!='train': return x
        if mode!='R': raise ValueError('This experiment accepts Right only')
        selection=self.plan[ids] if self.enabled else np.zeros(len(ids),dtype=np.int8)
        for row,sign in zip(ids,selection):
            self.trace.update(np.array([row,sign],dtype=np.int64).tobytes())
            self.counts[int(sign)]+=1
        for sign in [-1,1]:
            where=np.flatnonzero(selection==sign)
            if len(where):
                value=np.asarray(self.cache[sign][ids[where]])
                x[torch.as_tensor(where,device=device)]=torch.from_numpy(value).to(device=device,dtype=x.dtype)
        return x

def prepare_cache(output=None):
    out=Path(output or HERE/'cache'); out.mkdir(exist_ok=True)
    raw_path=BASE.parents[1]/'cache/X_clutterremoved.npy'
    source={'raw':_pipeline.digest(raw_path),
            'labels':_pipeline.digest(BASE/'prepared/y.npy'),'subjects':_pipeline.digest(BASE/'prepared/subj.npy'),
            'prepared':{n:_pipeline.digest(BASE/f'prepared/{n}.npy') for n in ['off','only']},
            'code':{str(p):_pipeline.digest(p) for p in [Path(__file__),BASE/'pipeline.py',BASE/'vendor/uwb_osr/data.py']},
            'numpy':np.__version__,'amounts':[-2,2],'fitted_parameters':False}
    manifest=out/'manifest.json'
    if manifest.exists():
        meta=json.loads(manifest.read_text(encoding='utf-8'))
        if meta['source']!=source: raise ValueError('Augmentation cache source changed')
        for name,h in meta['files'].items():
            if _pipeline.digest(out/name)!=h: raise ValueError('Augmentation cache content changed')
        return {-1:np.load(out/'minus2.npy',mmap_mode='r'),1:np.load(out/'plus2.npy',mmap_mode='r')},meta
    raw=np.load(raw_path,mmap_mode='r'); labels=np.load(BASE/'prepared/y.npy'); subjects=np.load(BASE/'prepared/subj.npy')
    for name,expected in [('y',labels),('subj',subjects)]:
        if not np.array_equal(np.load(raw_path.parent/f'{name}_clutterremoved.npy'),expected):
            raise ValueError('Original CIR and prepared metadata differ')
    amp=np.load(BASE/'prepared/off.npy',mmap_mode='r'); phase=np.load(BASE/'prepared/only.npy',mmap_mode='r')
    if raw.shape!=(len(labels),3,90,189): raise ValueError('Wrong original CIR layout')
    arrays={a:np.lib.format.open_memmap(out/f'{name}.partial.npy',mode='w+',dtype='float16',shape=(len(raw),1,90,378))
            for a,name in [(-2,'minus2'),(2,'plus2')]}
    max_error=0.; differences=0
    # Deterministic sample-wise preparation, no labels/statistics fitted here.
    # Training reader still enforces each fold's source IDs before accessing it.
    for start in range(0,len(raw),32):
        x=np.array(raw[start:start+32,2:3],copy=True)
        original=prepare_input(x,0)
        saved=np.concatenate([amp[start:start+32,2:3],phase[start:start+32,2:3]],axis=-1)
        err=np.abs(original.astype(float)-saved.astype(float))
        max_error=max(max_error,float(err.max())); differences+=int(np.count_nonzero(err))
        if max_error!=0: raise ValueError(f'Identity transform mismatch {max_error}')
        for a,array in arrays.items(): array[start:start+len(x)]=prepare_input(x,a)
        if start%1024==0: print('CACHE',start,'/',len(raw),flush=True)
    for array in arrays.values(): array.flush()
    arrays.clear(); del array
    for name in ['minus2','plus2']: (out/f'{name}.partial.npy').replace(out/f'{name}.npy')
    meta={'source':source,'identity_max_abs_error':max_error,'identity_differing_values':differences,
          'samples':len(raw),'files':{n:_pipeline.digest(out/n) for n in ['minus2.npy','plus2.npy']}}
    _pipeline.dump(manifest,meta)
    return {-1:np.load(out/'minus2.npy',mmap_mode='r'),1:np.load(out/'plus2.npy',mmap_mode='r')},meta

if __name__=='__main__':
    _,meta=prepare_cache(); print(json.dumps(meta,ensure_ascii=False,indent=2))
