"""Decompose ONE joint CE gradient through three shared-encoder view paths.

This is not three task losses or PCGrad. Fixed source-trained interval averages,
source TRAIN labels only, no optimizer updates or validation/test scoring.
"""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from audit_restored import digest, dump, quality_mask
from circular_signal import SignalNet, get_batch


def flat(g): return torch.cat([x.reshape(-1) for x in g])


def decompose(net,x,y):
    net.eval();params=list(net.encoder.parameters())
    z=net.encoder(x.reshape(-1,4,90,63)).reshape(len(x),3,-1)
    logits=net.head(z);loss=F.cross_entropy(logits,y)
    dz=torch.autograd.grad(loss,z,retain_graph=True)[0]
    total=flat(torch.autograd.grad(loss,params,retain_graph=True))
    contributions=[]
    for v in range(3):
        mask=torch.zeros_like(dz);mask[:,v]=dz[:,v]
        contributions.append(flat(torch.autograd.grad(z,params,grad_outputs=mask,retain_graph=v<2)))
    g=torch.stack(contributions);err=float(torch.linalg.vector_norm(g.sum(0)-total)/(torch.linalg.vector_norm(total)+1e-30))
    assert err<1e-5
    return g.detach(),total.detach(),logits.detach(),float(loss.detach()),err


def replica_check(net,x,y,g,logits):
    copies=[copy.deepcopy(net.encoder) for _ in range(3)]
    z=torch.stack([e(x[:,v]).flatten(1) for v,e in enumerate(copies)],1)
    output=net.head(z);loss=F.cross_entropy(output,y)
    refs=torch.stack([flat(torch.autograd.grad(loss,list(e.parameters()),retain_graph=v<2)) for v,e in enumerate(copies)])
    torch.testing.assert_close(output,logits,atol=2e-5,rtol=2e-5)
    relative=float(torch.linalg.vector_norm(g-refs)/(torch.linalg.vector_norm(refs)+1e-30))
    assert relative<2e-4
    return relative,float((output-logits).detach().abs().max())


def geometry(gram):
    norms=np.sqrt(np.maximum(np.diagonal(gram,axis1=-2,axis2=-1),0))
    valid=(norms>1e-20).all(-1)
    pairs=[(0,1),(0,2),(1,2)]
    cosine=np.stack([np.divide(gram[...,i,j],norms[...,i]*norms[...,j],out=np.zeros_like(norms[...,0]),where=valid) for i,j in pairs],-1)
    shares=np.divide(norms,norms.sum(-1,keepdims=True),out=np.zeros_like(norms),where=norms.sum(-1,keepdims=True)>0)
    cancel=np.divide(np.sqrt(np.maximum(gram.sum((-2,-1)),0)),norms.sum(-1),out=np.zeros_like(norms[...,0]),where=norms.sum(-1)>0)
    return {'cosine':cosine,'norm_share':shares,'cancellation_ratio':cancel,'valid':valid}


def summarize(rows,c):
    results={};conflict_folds=[];dominance_folds=[]
    for held in c['source_users']:
        out={};conflict_epochs=[];dominant_epochs=[]
        for epoch in c['epochs']:
            sub=[r for r in rows if r['held']==held and r['epoch']==epoch]
            geom=geometry(np.asarray([r['gram'] for r in sub]));v=geom['valid']
            conflict=(geom['cosine'].min(-1)<=c['screen']['cosine_max'])&v
            dominance=(geom['norm_share'].max(-1)>=c['screen']['dominant_share_min'])&v
            if conflict.mean()>=c['screen']['batch_fraction_min']: conflict_epochs.append(epoch)
            if dominance.mean()>=c['screen']['batch_fraction_min']: dominant_epochs.append(epoch)
            out[str(epoch)]={'batches':len(sub),'valid':int(v.sum()),'conflict_fraction':float(conflict.mean()),'dominance_fraction':float(dominance.mean()),
                'median_cosines_LT_LR_TR':np.median(geom['cosine'],axis=0).tolist(),'median_norm_shares_L_T_R':np.median(geom['norm_share'],axis=0).tolist(),
                'median_cancellation_ratio':float(np.median(geom['cancellation_ratio']))}
        if len(conflict_epochs)>=c['screen']['min_epochs']:conflict_folds.append(held)
        if len(dominant_epochs)>=c['screen']['min_epochs']:dominance_folds.append(held)
        results[str(held)]={'epochs':out,'conflict_epochs':conflict_epochs,'dominance_epochs':dominant_epochs}
    return {'trajectories':results,'conflict_trajectories':conflict_folds,'dominance_trajectories':dominance_folds,
        'screen_passed':len(conflict_folds)>=c['screen']['min_trajectories'] or len(dominance_folds)>=c['screen']['min_trajectories']}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ['data','prepared','models','out']:ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--protocol',type=Path,default=Path(__file__).resolve().parents[1]/'research/VIEW_GRADIENT_PROTOCOL.json')
    ap.add_argument('--cost-only',action='store_true');a=ap.parse_args();c=json.loads(a.protocol.read_text())
    if a.out.exists():raise ValueError('Use a fresh output directory')
    torch.set_num_threads(1)
    cp=a.prepared/'signals.npy';yp=a.data/'cache/y_clutterremoved.npy';up=a.data/'cache/subj_clutterremoved.npy'
    assert {k:digest(p) for k,p in [('cache',cp),('labels',yp),('users',up)]}==c['input_sha256']
    cache=np.load(cp,mmap_mode='r');y=np.load(yp);users=np.load(up);good=quality_mask(y,users)
    rows=[];samples={};models={};audit_grads={};start=time.perf_counter();rep_errors=[]
    for held in c['source_users']:
        folder=a.models/f'held{held}';split=json.loads((folder/'split.json').read_text())
        val=c['source_users'][(c['source_users'].index(held)+1)%6]
        train=np.flatnonzero(good&np.isin(users,[u for u in c['source_users'] if u not in [held,val]]))
        np.testing.assert_array_equal(train,split['train'])
        rng=np.random.default_rng(c['sample_seed']+held)
        selected=np.concatenate([rng.choice(train[users[train]==u],c['per_train_user'],replace=False) for u in sorted(set(users[train]))])
        samples[str(held)]=selected.tolist()
        for epoch in c['epochs']:
            path=folder/f'segment_{epoch:02d}.pt';sha=digest(path)
            historical=json.loads((folder/'result.json').read_text())
            assert sha==next(r['segment_sha256'] for r in historical['history'] if r['epoch']==epoch)
            record=torch.load(path,map_location='cpu',weights_only=False)
            assert record['epoch']==epoch and record['identity']['held']==held
            state={k.removeprefix('module.'):v for k,v in record['average']['weights'].items() if k.startswith('module.')}
            net=SignalNet();net.load_state_dict(state);net.double();net.eval()
            models[f'{held}_{epoch}']={'path':str(path),'sha256':sha,'kind':'within-epoch dense average; fixed epoch, not validation-selected model'}
            for b in range(0,len(selected),c['batch']):
                ids=selected[b:b+c['batch']];x=get_batch(cache,ids,'legacy').double();target=torch.as_tensor(y[ids],dtype=torch.long)
                t=time.perf_counter();g,total,logits,loss,err=decompose(net,x,target)
                decomposition_seconds=time.perf_counter()-t;replica_seconds=0.
                if b==0:
                    rt=time.perf_counter()
                    re,le=replica_check(net,x,target,g,logits);rep_errors.append({'held':held,'epoch':epoch,'gradient_relative_error':re,'logit_max_error':le})
                    replica_seconds=time.perf_counter()-rt
                    audit_grads[f'g_{held}_{epoch}']=g.numpy();audit_grads[f'total_{held}_{epoch}']=total.numpy()
                gram=g.double()@g.double().T
                rows.append({'held':held,'epoch':epoch,'batch_index':b//c['batch'],'source_user':int(users[ids[0]]),
                    'ids':ids.tolist(),'gram':gram.tolist(),'ce_train_eval':loss,'sum_relative_error':err,'seconds':time.perf_counter()-t})
                if a.cost_only:
                    estimated=decomposition_seconds*c['expected_batches']+replica_seconds*len(c['source_users'])*len(c['epochs'])
                    a.out.mkdir();dump(a.out/'COST.json',{'seconds_first_batch_including_replica':rows[-1]['seconds'],'decomposition_seconds':decomposition_seconds,'replica_seconds':replica_seconds,'estimated_seconds':estimated,'budget_seconds':c['cost_budget_seconds'],'passed':estimated<c['cost_budget_seconds'],'training_performed':False})
                    print(json.dumps({'cost_seconds':rows[-1]['seconds'],'estimated_seconds':estimated,'passed':estimated<c['cost_budget_seconds']}));return
        print(json.dumps({'completed_trajectory':held,'batches':len(rows)}),flush=True)
    assert len(rows)==c['expected_batches'];a.out.mkdir()
    np.savez_compressed(a.out/'first_batch_gradients.npz',**audit_grads)
    dump(a.out/'BATCHES.json',rows);dump(a.out/'SAMPLE_IDS.json',samples);dump(a.out/'MODEL_INDEX.json',models);dump(a.out/'protocol.json',c)
    summary=summarize(rows,c);summary.update(state='completed_source_training_gradient_diagnostic',batches=len(rows),models=len(models),
        total_seconds_excluding_input_hashes=time.perf_counter()-start,independent_replica_checks=rep_errors,
        max_sum_relative_error=max(r['sum_relative_error'] for r in rows),new_training=False,test_accuracy_computed=False,
        protocol_sha256=digest(a.protocol),script_sha256=digest(__file__),
        files_sha256={p.name:digest(p) for p in a.out.iterdir() if p.is_file()},
        limits=['eval-mode frozen interval averages; not online noisy training gradients','four source training users per trajectory, not diagnosis/validation users','views are paths of one objective, not independent PCGrad tasks','cancellation or dominance alone does not prove harmful optimization, generalization loss, or novelty'])
    dump(a.out/'SUMMARY.json',summary);print(json.dumps({'state':summary['state'],'screen_passed':summary['screen_passed'],'seconds':summary['total_seconds_excluding_input_hashes']}))


if __name__=='__main__':main()
