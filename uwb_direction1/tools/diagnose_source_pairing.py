"""Frozen source-training-pool clip re-pairing sensitivity, no fitting.

Separate per-view encoders permit exact evaluation via cached frozen features.
Every condition is checked against independently written raw-signal forward.
No validation/diagnostic label participates in pairing or scoring.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from audit_restored import digest,dump
from circular_signal import SignalNet,get_batch
from audit_circular_signal import independent_forward
from source_pairing import same_user_derangement,different_user_permutation


def train_ids(y,u,h):
    v=h+1 if h<8 else 3
    return np.flatnonzero((u>=3)&(u<=8)&(u!=h)&(u!=v)&~((u==6)&(y==3)))


def mean_groups(values,y,u):
    return float(np.mean([np.mean([values[(u==user)&(y==label)].mean() for label in np.unique(y[u==user])]) for user in np.unique(u)]))


def measurements(base,new,y,u):
    p=torch.softmax(torch.from_numpy(base).double(),1);q=torch.softmax(torch.from_numpy(new).double(),1);m=(p+q)/2
    js=(.5*(p*(p.log()-m.log())+q*(q.log()-m.log())).sum(1)).numpy()
    a=F.cross_entropy(torch.from_numpy(base).double(),torch.from_numpy(y).long(),reduction='none').numpy()
    b=F.cross_entropy(torch.from_numpy(new).double(),torch.from_numpy(y).long(),reduction='none').numpy()
    return dict(flip=mean_groups(base.argmax(1)!=new.argmax(1),y,u),js=mean_groups(js,y,u),ce_delta=mean_groups(b-a,y,u))


def independent_measurements(base,new,y,u):
    def prob(x):
        x=x.astype('float64');e=np.exp(x-x.max(1,keepdims=True));return e/e.sum(1,keepdims=True)
    p,q=prob(base),prob(new);m=(p+q)*.5
    vals=[(base.argmax(1)!=new.argmax(1)),.5*np.sum(p*np.log(p/m)+q*np.log(q/m),axis=1),np.log(p[np.arange(len(y)),y])-np.log(q[np.arange(len(y)),y])]
    weights=np.zeros(len(y))
    for user in set(u):
        classes=set(y[u==user])
        for cls in classes:
            mask=(u==user)&(y==cls);weights[mask]=1/(len(set(u))*len(classes)*sum(mask))
    return dict(zip(['flip','js','ce_delta'],[float(weights@v) for v in vals]))


@torch.no_grad()
def head(net,z,batch):return torch.cat([net.head(z[i:i+batch]) for i in range(0,len(z),batch)]).numpy()


@torch.no_grad()
def run(a,c):
    if a.out.exists():raise ValueError('Use new output directory')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    cp=a.prepared/'signals.npy';yp=a.data/'cache/y_clutterremoved.npy';up=a.data/'cache/subj_clutterremoved.npy'
    assert {k:digest(p) for k,p in [('cache',cp),('labels',yp),('users',up)]}==c['input_sha256']
    cache=np.load(cp,mmap_mode='r');y=np.load(yp);u=np.load(up);started=time.perf_counter();rows=[];models=[];audits=[]
    a.out.mkdir();dump(a.out/'protocol.json',c)
    for h in c['source_users']:
        ids=train_ids(y,u,h);folder=a.models/f'held{h}';split=json.loads((folder/'split.json').read_text());np.testing.assert_array_equal(ids,split['train'])
        path=folder/f"segment_{c['epoch']:02d}.pt";sha=digest(path);old=json.loads((folder/'result.json').read_text());assert sha==next(r['segment_sha256'] for r in old['history'] if r['epoch']==c['epoch'])
        record=torch.load(path,weights_only=False,map_location='cpu');assert record['epoch']==c['epoch'] and record['identity']['held']==h
        state={k.removeprefix('module.'):v for k,v in record['average']['weights'].items() if k.startswith('module.')};net=SignalNet();net.load_state_dict(state);net.eval()
        if a.cost_only:
            t=time.perf_counter();x=get_batch(cache,ids[:64],'legacy');z=net.encoder(x.reshape(-1,4,90,63)).reshape(len(x),3,-1);base=head(net,z,64);forward=time.perf_counter()-t
            t=time.perf_counter();ref=independent_forward(x,state).numpy();audit=time.perf_counter()-t;np.testing.assert_allclose(base,ref,atol=2e-5,rtol=2e-5)
            # 6 full source feature passes,15 head passes and15 independent64-record checks each,2x headroom.
            estimate=2*(6*4800/64*forward+6*15*audit+6*15*4800/64*.01)
            result=dict(feature_and_head_seconds64=forward,independent_seconds64=audit,estimated_seconds=estimate,budget_seconds=c['budget_seconds'],passed=estimate<c['budget_seconds'],no_sensitivity_metrics_computed=True)
            dump(a.out/'COST.json',result);print(json.dumps(result),flush=True);return
        z=torch.cat([net.encoder(get_batch(cache,ids[i:i+c['batch']],'legacy').reshape(-1,4,90,63)).reshape(len(ids[i:i+c['batch']]),3,-1) for i in range(0,len(ids),c['batch'])])
        base=head(net,z,c['batch']);position=np.full(len(y),-1,dtype=np.int64);position[ids]=np.arange(len(ids));save={'ids':ids,'source_y':y[ids],'source_users':u[ids],'original_logits':base};conditions=[]
        for seed in c['permutation_seeds']:
            same,detail=same_user_derangement(y,u,ids,seed+h);cross=different_user_permutation(y,u,ids,seed+h)
            assert detail['changed']==len(ids) and not detail['unchanged_singleton_ids']
            # Pool-external labels/users cannot influence either construction.
            yy=y.copy();uu=u.copy();outside=np.ones(len(y),bool);outside[ids]=False;yy[outside]=-99;uu[outside]=-99
            np.testing.assert_array_equal(same[ids],same_user_derangement(yy,uu,ids,seed+h)[0][ids]);np.testing.assert_array_equal(cross[ids],different_user_permutation(yy,uu,ids,seed+h)[ids])
            for kind,lookup in [('same',same),('cross',cross)]:
                targets=lookup[ids];assert set(targets)==set(ids) and np.all(targets!=ids);np.testing.assert_array_equal(y[targets],y[ids])
                assert np.all((u[targets]==u[ids]) if kind=='same' else (u[targets]!=u[ids]))
                save[f'{kind}_{seed}_target_ids']=targets
                for v in range(3):
                    mapping=np.repeat(ids[:,None],3,axis=1);mapping[:,v]=targets
                    conditions.append((f'{kind}_{seed}_v{v}',mapping,kind,seed,v))
        joint=np.random.default_rng(c['joint_seed']+h).permutation(ids);conditions.append(('joint',np.repeat(joint[:,None],3,axis=1),'joint',0,-1))
        conditions.append(('identity',np.repeat(ids[:,None],3,axis=1),'identity',0,-1))
        sample=np.sort(np.random.default_rng(c['audit_seed']+h).choice(len(ids),c['audit_samples'],replace=False));maxerr=0.
        for name,mapping,kind,seed,v in conditions:
            assert np.isin(mapping,ids).all();zz=torch.stack([z[position[mapping[:,j]],j] for j in range(3)],1);logits=head(net,zz,c['batch']);save[name+'_logits']=logits
            # Independent input gather and functional forward, not feature permutation.
            raw=np.asarray(cache[mapping[sample],np.arange(3)[None,:]],dtype=np.float32)
            x=torch.from_numpy(np.stack([raw[:,:,0],raw[:,:,2],np.zeros_like(raw[:,:,0]),np.zeros_like(raw[:,:,0])],2))
            ref=independent_forward(x,state).numpy();err=float(np.max(np.abs(ref-logits[sample])));maxerr=max(maxerr,err);np.testing.assert_allclose(logits[sample],ref,atol=2e-5,rtol=2e-5)
            if kind=='joint':np.testing.assert_allclose(logits,base[position[joint]],atol=2e-5,rtol=2e-5)
            elif kind=='identity':np.testing.assert_array_equal(logits,base)
            else:
                m=measurements(base,logits,y[ids],u[ids]);ind=independent_measurements(base,logits,y[ids],u[ids]);np.testing.assert_allclose(list(m.values()),list(ind.values()),atol=1e-10,rtol=1e-10)
                rows.append(dict(held=h,kind=kind,seed=seed,view=v,**m))
        save['joint_ids']=joint;save['audit_positions']=sample;np.savez_compressed(a.out/f'held{h}.npz',**save)
        models.append(dict(held=h,path=str(path),sha256=sha,samples=len(ids),kind='fixed epoch24 within-epoch dense average; no validation selection'))
        audits.append(dict(held=h,all_maps_source_only_bijective_class_preserving=True,outside_pool_labels_isolated=True,joint_reorder_passed=True,identity_bitwise_equal=True,raw_functional_conditions=len(conditions),samples_per_condition=len(sample),max_abs_logit_error=maxerr,independent_metrics_passed=True))
        print(json.dumps({'completed':h,'seconds':time.perf_counter()-started}),flush=True)
    screens=[]
    for v in range(3):
        passing=[];pairs=[]
        for h in c['source_users']:
            sr=[r for r in rows if r['held']==h and r['view']==v];ok=[]
            for seed in c['permutation_seeds']:
                same=next(r for r in sr if r['seed']==seed and r['kind']=='same')['flip'];cross=next(r for r in sr if r['seed']==seed and r['kind']=='cross')['flip'];ok.append(cross>=c['screen']['cross_flip_min'] and cross-same>=c['screen']['excess_flip_min']);pairs.append(dict(held=h,seed=seed,same_flip=same,cross_flip=cross,excess=cross-same))
            if all(ok):passing.append(h)
        screens.append(dict(view=v,passing_trajectories=passing,passed=len(passing)>=c['screen']['min_trajectories'],pairs=pairs))
    dump(a.out/'READINGS.json',rows);dump(a.out/'MODEL_INDEX.json',models);dump(a.out/'AUDIT.json',dict(state='passed',trajectories=audits));
    dump(a.out/'SUMMARY.json',dict(state='completed',screen_passed=any(s['passed'] for s in screens),screens=screens,seconds_excluding_hashes=time.perf_counter()-started,new_training=False,validation_or_diagnostic_scoring=False,protocol_sha256=digest(a.protocol),script_sha256=digest(__file__),files_sha256={p.name:digest(p) for p in a.out.iterdir() if p.is_file()},limits=c['limits']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['data','prepared','models','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=Path(__file__).resolve().parents[1]/'research/SOURCE_PAIRING_PROTOCOL.json');p.add_argument('--cost-only',action='store_true');a=p.parse_args();run(a,json.loads(a.protocol.read_text()))
if __name__=='__main__':main()
