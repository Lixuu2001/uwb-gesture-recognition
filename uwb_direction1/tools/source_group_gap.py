"""Source train/validation group-risk diagnosis; no held-user signal or score."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp
from circular_signal import SignalNet,get_batch
from audit_circular_signal import independent_forward
from audit_restored import digest,dump


def statistics(logits,y,u):
    ce=logsumexp(logits.astype('float64'),axis=1)-logits[np.arange(len(y)),y];wrong=logits.argmax(1)!=y
    groups={}
    for user in np.unique(u):
        classes={}
        for label in np.unique(y[u==user]):
            mask=(u==user)&(y==label);classes[str(label)]=dict(n=int(mask.sum()),ce=float(ce[mask].mean()),error=float(wrong[mask].mean()))
        groups[str(user)]=dict(classes=classes,macro_ce=float(np.mean([v['ce'] for v in classes.values()])),macro_error=float(np.mean([v['error'] for v in classes.values()])))
    return groups


def main(a):
    c=json.loads(a.protocol.read_text());torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if a.out.exists():raise ValueError('Fresh output directory required')
    base=Path(c['data'])/'cache';paths={'cache':Path(c['prepared'])/'signals.npy','labels':base/'y_clutterremoved.npy','users':base/'subj_clutterremoved.npy'}
    assert {k:digest(v) for k,v in paths.items()}==c['input_sha256'];cache=np.load(paths['cache'],mmap_mode='r');y=np.load(paths['labels']);u=np.load(paths['users']);a.out.mkdir();dump(a.out/'protocol.json',c);rows=[];models=[];start=time.perf_counter();independent=[]
    for h in c['source_users']:
        modeldir=Path(c['models'])/f'held{h}';old=json.loads((modeldir/'result.json').read_text());split=json.loads((modeldir/'split.json').read_text());v=h+1 if h<8 else 3
        good=(u>=3)&(u<=8)&~((u==6)&(y==3));parts=dict(train=np.flatnonzero(good&(u!=h)&(u!=v)),val=np.flatnonzero(good&(u==v)))
        for role,ids in parts.items():np.testing.assert_array_equal(ids,split[role]);assert not np.any(u[ids]==h)
        d=a.out/f'held{h}';d.mkdir();dump(d/'roles.json',{k:ids.tolist() for k,ids in parts.items()})
        for epoch in c['epochs']:
            p=modeldir/f'segment_{epoch:02d}.pt';sha=digest(p);assert sha==next(x['segment_sha256'] for x in old['history'] if x['epoch']==epoch)
            checkpoint=torch.load(p,weights_only=False,map_location='cpu');assert checkpoint['identity']['held']==h and checkpoint['epoch']==epoch
            state={k.removeprefix('module.'):t for k,t in checkpoint['average']['weights'].items() if k.startswith('module.')};net=SignalNet();net.load_state_dict(state);net.eval()
            if a.cost:
                ids=parts['train'][:128];t=time.perf_counter()
                with torch.no_grad():
                    for i in range(0,len(ids),c['batch']):net(get_batch(cache,ids[i:i+c['batch']],'legacy'))
                secs=time.perf_counter()-t;estimate=2*18*6000/len(ids)*secs
                dump(a.out/'COST.json',dict(seconds128=secs,estimate_seconds=estimate,budget_seconds=c['budget_seconds'],passed=estimate<c['budget_seconds'],source_train_ids=ids.tolist(),quality_scored=False));print(json.dumps(dict(estimate_seconds=estimate,passed=estimate<c['budget_seconds'])));return
            pred={};entry=dict(held=h,epoch=epoch,roles={})
            for role,ids in parts.items():
                with torch.no_grad():z=torch.cat([net(get_batch(cache,ids[i:i+c['batch']],'legacy')) for i in range(0,len(ids),c['batch'])]).numpy()
                # Independent tensor-level forward for first and last full/partial batch.
                for i in [0,((len(ids)-1)//c['batch'])*c['batch']]:
                    ref=independent_forward(get_batch(cache,ids[i:i+c['batch']],'legacy'),state).numpy();err=float(abs(ref-z[i:i+c['batch']]).max());assert err<2e-5;independent.append(dict(held=h,epoch=epoch,role=role,start=i,max_error=err))
                pred[role+'_ids']=ids;pred[role+'_logits']=z;entry['roles'][role]=statistics(z,y[ids],u[ids])
            np.savez_compressed(d/f'epoch{epoch:02d}.npz',**pred);rows.append(entry);models.append(dict(held=h,epoch=epoch,path=str(p),sha256=sha))
        print(json.dumps(dict(held=h,complete=True,held_input_read=False)),flush=True)
    dump(a.out/'GROUPS.json',rows);dump(a.out/'MODELS.json',models);dump(a.out/'FORWARD_AUDIT.json',independent)
    qualify=[];stages=[]
    for epoch in c['epochs']:
        ee=[r for r in rows if r['epoch']==epoch];trainmax=[];valerr=[];traince=[];valce=[]
        for r in ee:
            train=list(r['roles']['train'].values());val=list(r['roles']['val'].values());trainmax.append(max(x['macro_error'] for x in train));valerr.append(val[0]['macro_error']);traince.append(max(x['macro_ce'] for x in train));valce.append(val[0]['macro_ce'])
        stages.append(dict(epoch=epoch,train_worst_error=trainmax,val_error=valerr,train_worst_ce=traince,val_ce=valce))
    for h in c['source_users']:
        passes=[]
        for epoch in c['gate']['epochs']:
            r=next(r for r in rows if r['held']==h and r['epoch']==epoch);errs=[v['macro_error'] for v in r['roles']['train'].values()]
            passes.append(max(errs)>=c['gate']['worst_train_error_min'] and max(errs)-min(errs)>=c['gate']['train_error_gap_min'])
        if all(passes):qualify.append(h)
    files={str(p.relative_to(a.out)):digest(p) for p in a.out.rglob('*') if p.is_file()}
    summary=dict(seconds=time.perf_counter()-start,stages=stages,qualifying_trajectories=qualify,persistent_training_group_gate=len(qualify)>=c['gate']['min_trajectories'],models=18,source_role_rows=sum(len(parts) for h in c['source_users'] for parts in json.loads((a.out/f'held{h}/roles.json').read_text()).values())*len(c['epochs']),files=files,scope='Source train and validation roles only; six overlapping trajectories, not independent samples. No held classification, training, reweighting, or novel method result.')
    dump(a.out/'SUMMARY.json',summary);print(json.dumps({k:summary[k] for k in ['seconds','models','source_role_rows','qualifying_trajectories','persistent_training_group_gate']}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cost',action='store_true');main(p.parse_args())
