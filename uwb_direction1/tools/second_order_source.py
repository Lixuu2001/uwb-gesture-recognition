"""Fixed source-encoder second-order readout pilot. Not end-to-end MPN-COV."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from audit_restored import digest,dump
from circular_signal import SignalNet,get_batch
from audit_circular_signal import independent_forward
from second_order_core import projection,describe,weights,fit_family,score,macro


def roles(y,u,h):
    good=(u>=3)&(u<=8)&~((u==6)&(y==3));v=h+1 if h<8 else 3
    return dict(train=np.flatnonzero(good&(u!=h)&(u!=v)),val=np.flatnonzero(good&(u==v)),diagnostic=np.flatnonzero(good&(u==h)))


@torch.no_grad()
def extract(net,cache,ids,p,batch):
    arrays={k:[] for k in ['grid','diag','cov','sqrt','cnn']}
    for i in range(0,len(ids),batch):
        x=get_batch(cache,ids[i:i+batch],'legacy');z=net.encoder[:3](x.reshape(-1,4,90,63));grid=F.adaptive_avg_pool2d(z,(3,4)).reshape(len(x),-1)
        arrays['grid'].append(grid.numpy().astype('float64'));arrays['cnn'].append(net.head(grid).numpy())
        for k,a in describe(z.reshape(len(x),3,48,-1).numpy(),p).items():arrays[k].append(a)
    return {k:np.concatenate(a) for k,a in arrays.items()}


@torch.no_grad()
def audit_descriptors(x,state,p,features):
    from scipy.linalg import sqrtm
    n=len(x);z=x.reshape(-1,4,90,63)
    for stage in range(3):
        b=f'encoder.{stage}.layers.'
        z=F.gelu(F.group_norm(F.conv2d(z,state[b+'0.weight'],stride=2,padding=1),4,state[b+'1.weight'],state[b+'1.bias']))
        z=F.gelu(F.group_norm(F.conv2d(z,state[b+'3.weight'],padding=1),4,state[b+'4.weight'],state[b+'4.bias']))
    grid=F.adaptive_avg_pool2d(z,(3,4)).reshape(n,-1).numpy();np.testing.assert_array_equal(grid,features['grid'][:n])
    maps=z.reshape(n,3,48,-1).numpy().astype('float64');i,j=np.triu_indices(16);w=np.where(i==j,1.,np.sqrt(2));errs=[]
    for sample in range(n):
        for view in range(3):
            a=p.T@maps[sample,view];cov=np.cov(a,bias=True);root=sqrtm(cov)
            if np.iscomplexobj(root):assert abs(root.imag).max()<1e-7;root=root.real
            refs={'diag':np.r_[a.mean(1),np.sqrt(np.maximum(cov.diagonal(),0))], 'cov':np.r_[a.mean(1),cov[i,j]*w], 'sqrt':np.r_[a.mean(1),root[i,j]*w]}
            for arm,ref in refs.items():
                got=features[arm][sample].reshape(3,-1)[view];np.testing.assert_allclose(got,ref,atol=1e-7,rtol=1e-7);errs.append(float(abs(got-ref).max()))
    return max(errs)


def main(a,c):
    if a.out.exists():raise ValueError('Use fresh output directory')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    cp=a.prepared/'signals.npy';yp=a.data/'cache/y_clutterremoved.npy';up=a.data/'cache/subj_clutterremoved.npy'
    assert {k:digest(p) for k,p in [('cache',cp),('labels',yp),('users',up)]}==c['input_sha256']
    cache=np.load(cp,mmap_mode='r');y=np.load(yp);u=np.load(up);p=projection(c['projection_seed']);a.out.mkdir();dump(a.out/'protocol.json',c)
    np.save(a.out/'projection.npy',p);started=time.perf_counter();results=[]
    for h in c['source_users']:
        parts=roles(y,u,h);folder=a.models/f'held{h}';old=json.loads((folder/'result.json').read_text());sp=json.loads((folder/'split.json').read_text())
        for k,ids in parts.items():np.testing.assert_array_equal(ids,sp[k])
        checkpoint=folder/f"segment_{c['epoch']:02d}.pt";sha=digest(checkpoint);assert sha==next(r['segment_sha256'] for r in old['history'] if r['epoch']==c['epoch'])
        record=torch.load(checkpoint,weights_only=False,map_location='cpu');assert record['epoch']==c['epoch'] and record['identity']['held']==h
        state={k.removeprefix('module.'):v for k,v in record['average']['weights'].items() if k.startswith('module.')};net=SignalNet();net.load_state_dict(state);net.eval()
        if a.cost_only:
            t=time.perf_counter();f=extract(net,cache,parts['train'][:64],p,64);secs=time.perf_counter()-t
            # Full-dimension synthetic solve checks runtime without seeing validation/diagnostic labels.
            rng=np.random.default_rng(44);x=rng.normal(size=(4800,1728));yy=np.arange(4800)%12;uu=np.arange(4800)%4
            t=time.perf_counter();fit_family(x,yy,uu,c['lambdas']);fitsecs=time.perf_counter()-t
            estimate=2*(6*7100/64*secs+6*4*fitsecs*2)
            dump(a.out/'COST.json',dict(feature_seconds64=secs,synthetic_full_dimension_fit_seconds=fitsecs,estimated_seconds=estimate,budget_seconds=c['budget_seconds'],passed=estimate<c['budget_seconds'],real_validation_or_diagnostic_scored=False));print(json.dumps({'estimated_seconds':estimate,'passed':estimate<c['budget_seconds']}),flush=True);return
        d=a.out/f'held{h}';d.mkdir();dump(d/'split.json',{k:v.tolist() for k,v in parts.items()})
        # Extraction is per sample; fitting accepts ONLY source arrays. No cross-role normalization.
        features={k:extract(net,cache,ids,p,c['batch']) for k,ids in parts.items()}
        raw_errors={}
        for k,ids in parts.items():
            ref=independent_forward(get_batch(cache,ids[:32],'legacy'),state).numpy();err=float(abs(ref-features[k]['cnn'][:32]).max());assert err<2e-5;raw_errors[k]=dict(logits=err,descriptors=audit_descriptors(get_batch(cache,ids[:32],'legacy'),state,p,features[k]))
        pred={'ids':parts['diagnostic'],'y':y[parts['diagnostic']],'validation_ids':parts['val'],'validation_y':y[parts['val']],'cnn':features['diagnostic']['cnn']};model={};selection={};audits=[]
        # All readout selections finish before diagnostic labels are used for metrics.
        for arm in c['arms']:
            train=parts['train'];val=parts['val'];x=features['train'][arm];m,s,b,coefs=fit_family(x,y[train],u[train],c['lambdas']);candidates=[]
            for lam,coef in zip(c['lambdas'],coefs):
                vp=score(features['val'][arm],m,s,b,coef);acc=macro(y[val],vp.argmax(1));mse=float(np.mean((vp-np.eye(12)[y[val]])**2));candidates.append(dict(lam=lam,macro=acc,mse=mse))
                pred[f'{arm}_validation_{lam}']=vp
                ref=Ridge(alpha=lam*len(train),solver='cholesky').fit((x-m)/s,np.eye(12)[y[train]],sample_weight=weights(y[train],u[train])*len(train))
                err=float(np.max(np.abs(vp-ref.predict((features['val'][arm]-m)/s))));assert err<1e-7;audits.append(dict(arm=arm,lam=lam,sklearn_max_error=err))
            choice=min(range(len(candidates)),key=lambda j:(-candidates[j]['macro'],candidates[j]['mse'],-candidates[j]['lam']));selection[arm]=dict(selected=choice,candidates=candidates,dimension=x.shape[1]);coef=coefs[choice]
            for k,z in [('mean',m),('scale',s),('bias',b),('coef',coef)]:model[f'{arm}_{k}']=z
            pred[arm]=score(features['diagnostic'][arm],m,s,b,coef)
        dump(d/'selection.json',selection);np.savez_compressed(d/'models.npz',**model)
        # Historical strong tail comparator, integrity and IDs checked; never used in selection.
        assert digest(folder/'predictions.npz')==old['files']['predictions.npz'];tail=np.load(folder/'predictions.npz');np.testing.assert_array_equal(tail['ids'],parts['diagnostic']);pred['tail_cnn']=tail['tail']
        scores={k:macro(pred['y'],pred[k].argmax(1)) for k in c['arms']+['cnn','tail_cnn']};np.savez_compressed(d/'predictions.npz',**pred)
        row=dict(held=h,scores=scores,model_path=str(checkpoint),model_sha256=sha,independent_raw_forward_max_errors=raw_errors,independent_ridge_fits=audits,files={n:digest(d/n) for n in ['split.json','selection.json','models.npz','predictions.npz']});dump(d/'result.json',row);results.append(row)
        print(json.dumps({'completed':h,'seconds':time.perf_counter()-started}),flush=True)
    scores={arm:[r['scores'][arm] for r in results] for arm in c['arms']+['cnn','tail_cnn']};candidate=np.array(scores['sqrt']);comparisons={}
    for arm,delta in c['gate']['min_mean_gain'].items():
        base=np.array(scores[arm]);diff=candidate-base;comparisons[arm]=dict(mean_delta=float(diff.mean()),wins=int((diff>1e-12).sum()),worst_user_delta=float(candidate.min()-base.min()),passed=bool(diff.mean()>=delta and (diff>1e-12).sum()>=c['gate']['min_wins'] and candidate.min()-base.min()>=-c['gate']['max_worst_decline']))
    dump(a.out/'SUMMARY.json',dict(state='complete',scores=scores,means={k:float(np.mean(v)) for k,v in scores.items()},comparisons=comparisons,gate_passed=all(v['passed'] for v in comparisons.values()),seconds_excluding_hashes=time.perf_counter()-started,selected_models=24,ridge_fits=72,encoder_training=False,protocol_sha256=digest(a.protocol),script_sha256=digest(__file__),helpers_sha256={n:digest(Path(__file__).with_name(n)) for n in ['second_order_core.py','circular_signal.py','audit_circular_signal.py','audit_restored.py']},rows=results))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ['data','prepared','models','out']:ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--protocol',type=Path,default=Path(__file__).resolve().parents[1]/'research/SECOND_ORDER_PROTOCOL.json');ap.add_argument('--cost-only',action='store_true');a=ap.parse_args()
    with threadpool_limits(limits=1):main(a,json.loads(a.protocol.read_text()))
