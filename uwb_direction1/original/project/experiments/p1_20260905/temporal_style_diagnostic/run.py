"""Frozen-model diagnostic. Outer test data are never loaded."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from core import time_map, warp, renorm, dtw_map, distance, descriptors, style_mix, partners

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE/'cross_user_meta'))
from model import Classifier
from protocol import open_fold, dump, digest

CONFIG={'sync':'user-authorized assumption: all three views frame aligned; not verified',
        'seed':20260908,'batch':32,'per_source_user_class':2,'warp_amplitudes':[-4,4],
        'shift_frames':[-3,3],'style_donor_shares':[.1,.3],'dtw_band':12,
        'evaluation_frames':[5,84],'checkpoint':'warmup/right.pt',
        'outer_test_access':False,'no_new_training':True}

def array(x): return x.detach().float().cpu().numpy()

def rel(a,b):
    a=np.asarray(a,dtype=np.float64).reshape(len(a),-1)
    b=np.asarray(b,dtype=np.float64).reshape(len(b),-1)
    return np.linalg.norm(b-a,axis=1)/np.maximum(np.linalg.norm(a,axis=1),1e-12)

class Observe:
    def __init__(self,model):
        self.data={}; self.donor=None; self.share=0; self.handles=[]
        for name in ['temporal','spatial']:
            path=getattr(model.encoder,name)
            def project(module,args,out,key=name):
                if self.donor is not None: out=style_mix(out,self.donor[key+'_proj'],self.share)
                self.data[key+'_proj']=out.detach().clone()
                return out
            self.handles.append(path.proj.register_forward_hook(project))
            block=path.encoder.layers[0] if hasattr(path.encoder,'layers') else path.encoder[0]
            norm=block.norm1 if hasattr(block,'norm1') else block.n1
            def capture(module,args,out,key=name): self.data[key+'_norm']=out.detach().clone()
            self.handles.append(norm.register_forward_hook(capture))
            def final(module,args,out,key=name): self.data[key+'_final']=out.detach().clone()
            self.handles.append(path.register_forward_hook(final))
    def clear(self): self.data={}; self.donor=None; self.share=0
    def close(self):
        for h in self.handles: h.remove()

@torch.no_grad()
def infer(model,obs,x,donor=None,share=0):
    obs.clear(); obs.donor=donor; obs.share=share
    with sdpa_kernel(SDPBackend.MATH),torch.autocast('cuda',dtype=torch.bfloat16):
        logits=model(x)[0]
    return array(logits),dict(obs.data)

def classify_fold(model,reader,y,parts,fold,out,obs):
    ids=parts['val']; train=parts['train']; rng=np.random.default_rng(CONFIG['seed']+fold)
    # Donor choice depends only on training IDs and seed, never validation labels.
    donor_ids=rng.choice(train,size=len(ids),replace=True)
    records={'val_ids':ids,'labels':y[ids],'donor_ids':donor_ids}
    values={}; changes={}
    def collect(key,logits): values.setdefault(key,[]).append(logits)
    for start in range(0,len(ids),32):
        sl=slice(start,start+32)
        x=reader.load_ids(ids[sl],'val','R')
        z,clean=infer(model,obs,x); collect('clean',z)
        z,_=infer(model,obs,renorm(x)); collect('renorm_control',z)
        for amount in CONFIG['warp_amplitudes']:
            z,_=infer(model,obs,renorm(warp(x,time_map(90,amount))))
            collect('warp_'+str(amount),z)
        for amount in CONFIG['shift_frames']:
            z,_=infer(model,obs,renorm(warp(x,np.arange(90)+amount)))
            collect('shift_'+str(amount),z)
        d=reader.load_ids(donor_ids[sl],'train','R')
        _,donor=infer(model,obs,d)
        for share in CONFIG['style_donor_shares']:
            key='style_'+str(share)
            z,modified=infer(model,obs,x,donor=donor,share=share); collect(key,z)
            for layer in clean:
                changes.setdefault(key+'_'+layer,[]).append(rel(array(clean[layer]),array(modified[layer])))
        if start%320==0: print('fold',fold,'validation',start,'/',len(ids),flush=True)
    records.update({k+'_logits':np.concatenate(v) for k,v in values.items()})
    records.update({k:np.concatenate(v) for k,v in changes.items()})
    saved=BASE/'cross_user_meta/runs_s2026'/f'fold{fold}/warmup/predictions.npz'
    with np.load(saved) as p:
        assert np.array_equal(p['val_ids'],ids)
        error=float(np.max(np.abs(p['val_logits']-records['clean_logits'])))
        assert error==0, f'Baseline replay mismatch {error}'
    np.savez_compressed(out/'validation.npz',**records)
    return {'samples':len(ids),'baseline_max_abs_error':error,'donors_training_only':bool(np.isin(donor_ids,train).all())}

def selected_ids(parts,y,s,fold):
    rng=np.random.default_rng(CONFIG['seed']+100+fold); ids=[]
    for u in np.unique(s[parts['train']]):
        for c in np.unique(y[parts['train'][s[parts['train']]==u]]):
            group=parts['train'][(s[parts['train']]==u)&(y[parts['train']]==c)]
            ids.extend(rng.choice(group,CONFIG['per_source_user_class'],replace=False))
    return np.sort(ids)

def alignment_fold(model,reader,y,s,parts,fold,out,obs):
    ids=selected_ids(parts,y,s,fold)
    other=partners(ids,y,s,CONFIG['seed']+fold,True)
    mismatch=partners(ids,y,s,CONFIG['seed']+fold,False)
    index={int(x):k for k,x in enumerate(ids)}
    raw=[]; original_tokens=[]; warped_tokens={a:[] for a in CONFIG['warp_amplitudes']}
    for start in range(0,len(ids),32):
        x=reader.load_ids(ids[start:start+32],'train','MV')
        raw.append(array(x))
        _,state=infer(model,obs,x[:,2:3]); original_tokens.append(array(state['temporal_final']))
        for a in CONFIG['warp_amplitudes']:
            _,state=infer(model,obs,renorm(warp(x[:,2:3],time_map(90,a))))
            warped_tokens[a].append(array(state['temporal_final']))
    raw=np.concatenate(raw); h=np.concatenate(original_tokens)
    desc=descriptors(raw); rows=[]; equiv=[]; maps=[]
    for a in CONFIG['warp_amplitudes']:
        g=time_map(90,a); hw=np.concatenate(warped_tokens[a])
        descw=descriptors(renorm(warp(torch.from_numpy(raw),g)).numpy())
        for k,idx in enumerate(ids):
            bad=index[int(mismatch[k])]; j=index[int(other[k])]
            costs=[distance(desc[k,v],descw[k,v]) for v in range(3)]
            methods={'position':np.arange(90,dtype=float),
                     'raw_R':dtw_map(costs[2]),'raw_MV':dtw_map(np.mean(costs,axis=0)),
                     'raw_MV_mismatch':dtw_map((costs[2]+distance(desc[k,0],descw[bad,0])+distance(desc[k,1],descw[bad,1]))/3),
                     'learned_R':dtw_map(distance(h[k],hw[k]))}
            for method,p in methods.items():
                err=np.abs(np.interp(p,np.arange(90),g)-np.arange(90))[5:-5]
                rows.append({'id':int(idx),'user':int(s[idx]),'class':int(y[idx]),'amplitude':a,
                             'method':method,'mae_frames':float(err.mean()),'within2':float(np.mean(err<=2))})
                maps.append(p)
            # Cross-user equivariance: compare matching under known deformation,
            # not with any claimed semantic ground truth.
            original=[distance(desc[k,v],desc[j,v]) for v in range(3)]
            changed=[distance(desc[k,v],descw[j,v]) for v in range(3)]
            for method,c0,c1 in [('raw_R',original[2],changed[2]),
                                 ('raw_MV',np.mean(original,axis=0),np.mean(changed,axis=0)),
                                 ('learned_R',distance(h[k],h[j]),distance(h[k],hw[j]))]:
                p0=dtw_map(c0); p1=dtw_map(c1)
                err=np.abs(np.interp(p1,np.arange(90),g)-p0)[5:-5]
                equiv.append({'id':int(idx),'partner':int(other[k]),'amplitude':a,'method':method,
                              'mae_frames':float(err.mean()),'path_deviation_frames':float(np.abs(p0-np.arange(90))[5:-5].mean())})
            if k%48==0: print('fold',fold,'alignment',a,k,'/',len(ids),flush=True)
    np.savez_compressed(out/'alignment_inputs.npz',ids=ids,cross_user_partner=other,
                        mismatch_partner=mismatch,descriptors=desc,learned_temporal=h,
                        **{'warped_temporal_'+str(a):np.concatenate(warped_tokens[a]) for a in warped_tokens},
                        synthetic_maps=np.asarray(maps))
    dump(out/'alignment_rows.json',{'synthetic':rows,'cross_user_equivariance':equiv})
    return {'selected_source_samples':len(ids),'only_train_ids':bool(np.isin(ids,parts['train']).all()),
            'mismatch_no_self':bool(np.all(ids!=mismatch)),
            'mismatch_same_user_class':bool(np.all(s[ids]==s[mismatch]) and np.all(y[ids]==y[mismatch])),
            'cross_user_same_class':bool(np.all(s[ids]!=s[other]) and np.all(y[ids]==y[other]))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--folds',nargs='+',type=int,default=list(range(1,9)))
    ap.add_argument('--output',default='runs'); args=ap.parse_args()
    root=HERE/args.output; root.mkdir(exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(CONFIG['seed'])
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    hashes={str(p.relative_to(BASE)):digest(p) for p in [HERE/'core.py',HERE/'run.py',HERE/'PLAN.md',
            BASE/'cross_user_meta/model.py',BASE/'cross_user_meta/protocol.py',BASE/'vendor/uwb_osr/model.py',BASE/'train.py']}
    dump(root/'manifest.json',{'config':CONFIG,'source_hashes':hashes,'folds':args.folds})
    for fold in args.folds:
        out=root/f'fold{fold}'; out.mkdir(exist_ok=True)
        if (out/'complete.json').exists(): raise RuntimeError('Output already complete; refusing overwrite')
        start=time.time(); reader,y,s,parts=open_fold(fold)
        checkpoint=BASE/'cross_user_meta/runs_s2026'/f'fold{fold}/warmup/right.pt'
        model=Classifier([2],'concat','mean').cuda().eval()
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False),strict=True)
        obs=Observe(model)
        val=classify_fold(model,reader,y,parts,fold,out,obs)
        align=alignment_fold(model,reader,y,s,parts,fold,out,obs)
        obs.close(); del model; torch.cuda.empty_cache()
        dump(out/'complete.json',{'fold':fold,'validation_user':fold%8+1,'model_hash':digest(checkpoint),
                                'validation':val,'alignment':align,'seconds':time.time()-start})
        print('FOLD_COMPLETE',fold,round(time.time()-start,1),flush=True)
    print('DIAGNOSTICS_COMPLETE',flush=True)

if __name__=='__main__': main()
