"""Preregistered source-only frozen-CNN plus range-energy area readout screen."""
import argparse,json,time,platform
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from audit_restored import digest,dump
from circular_signal import SignalNet,get_batch
from second_order_core import fit_family,score,macro,weights
from energy_area import energy_path,describe,source_roles,arm_features,decision


@torch.no_grad()
def extract(net,cache,ids):
    hidden=[];logits=[]
    for start in range(0,len(ids),64):
        x=get_batch(cache,ids[start:start+64],'legacy');g=net.encoder(x.reshape(-1,4,90,63)).reshape(len(x),-1)
        z=net.head[:3](g);hidden.append(z.numpy());logits.append(net.head[3:](z).numpy())
    return np.concatenate(hidden).astype(np.float64),np.concatenate(logits)


def load_net(folder):
    result=json.loads((folder/'result.json').read_text());p=folder/'segment_24.pt'
    sha=digest(p);assert sha==next(r['segment_sha256'] for r in result['history'] if r['epoch']==24)
    record=torch.load(p,weights_only=False,map_location='cpu')
    state={k.removeprefix('module.'):v for k,v in record['average']['weights'].items() if k.startswith('module.')}
    net=SignalNet();net.load_state_dict(state);net.eval();return net,sha,result


def main(a):
    c=json.loads(a.protocol.read_text());assert not a.out.exists();a.out.mkdir()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    data=a.data/'cache';raw=np.load(data/'X_clutterremoved.npy',mmap_mode='r');y=np.load(data/'y_clutterremoved.npy');u=np.load(data/'subj_clutterremoved.npy')
    cache=np.load(a.prepared/'signals.npy',mmap_mode='r')
    paths={'raw':data/'X_clutterremoved.npy','labels':data/'y_clutterremoved.npy','users':data/'subj_clutterremoved.npy','prepared':a.prepared/'signals.npy'}
    hashes={k:digest(v) for k,v in paths.items()};assert hashes==c['input_sha256']
    identity={'protocol_sha256':digest(a.protocol),'inputs':hashes,'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
      'code_sha256':{n:digest(Path(__file__).with_name(n)) for n in ['energy_area.py','run_energy_area.py','second_order_core.py','circular_signal.py','audit_restored.py']}}
    dump(a.out/'IDENTITY.json',identity);dump(a.out/'protocol.json',c)
    ids=np.flatnonzero(np.isin(u,range(3,9))&~((u==6)&(y==3)));assert len(ids)==7100
    if a.cost_only:
        train=source_roles(y,u,3)['train'][:64];net,_,_=load_net(a.models/'held3');t=time.perf_counter();h,_=extract(net,cache,train);cnnsecs=time.perf_counter()-t
        t=time.perf_counter();d=describe(energy_path(raw[train]),train,c['shuffle_seed']);descsecs=time.perf_counter()-t
        rng=np.random.default_rng(72);synthetic=rng.normal(size=(4800,254));t=time.perf_counter()
        fit_family(synthetic,np.arange(4800)%12,np.arange(4800)%4,c['lambdas']);fitsecs=time.perf_counter()-t
        estimate=2*(6*7100/64*cnnsecs+7100/64*descsecs+6*5*4*fitsecs)
        cost=dict(cnn_seconds64=cnnsecs,descriptor_seconds64=descsecs,synthetic_fit_seconds=fitsecs,estimated_seconds=estimate,budget_seconds=c['budget_seconds'],passed=estimate<c['budget_seconds'],no_validation_or_diagnostic_scores=True)
        dump(a.out/'COST.json',cost);print(json.dumps(cost),flush=True);return
    cost=json.loads(a.cost.read_text());assert cost['passed'];assert json.loads((a.cost.parent/'IDENTITY.json').read_text())['protocol_sha256']==identity['protocol_sha256']
    started=time.perf_counter();desc={k:[] for k in ['base','signed','unsigned','shuffled']}
    for start in range(0,len(ids),64):
        ii=ids[start:start+64];z=describe(energy_path(raw[ii]),ii,c['shuffle_seed'])
        for k in desc:desc[k].append(z[k])
    desc={k:np.concatenate(v) for k,v in desc.items()};np.savez_compressed(a.out/'descriptors.npz',ids=ids,**desc)
    lookup=np.full(len(y),-1);lookup[ids]=np.arange(len(ids));scores={k:[] for k in c['arms']+['cnn','tail_cnn']};rows=[]
    for held in c['source_users']:
        d=a.out/f'held{held}';d.mkdir();parts=source_roles(y,u,held);oldfolder=a.models/f'held{held}'
        oldsplit=json.loads((oldfolder/'split.json').read_text())
        for k,ii in parts.items():np.testing.assert_array_equal(ii,oldsplit[k])
        net,sha,old=load_net(oldfolder);hidden,cnn=extract(net,cache,ids);np.savez_compressed(d/'features.npz',ids=ids,hidden=hidden,cnn=cnn)
        features=arm_features(hidden,desc);sel={};models={};pred={'ids':parts['diagnostic'],'y':y[parts['diagnostic']],'validation_ids':parts['val'],'validation_y':y[parts['val']]};checks=[]
        tr=parts['train'];va=parts['val'];te=parts['diagnostic'];it=lookup[tr];iv=lookup[va];ie=lookup[te]
        for arm in c['arms']:
            x=features[arm];mean,scale,bias,coefs=fit_family(x[it],y[tr],u[tr],c['lambdas']);candidates=[]
            for lam,coef in zip(c['lambdas'],coefs):
                vp=score(x[iv],mean,scale,bias,coef);pred[f'{arm}_validation_{lam}']=vp
                candidates.append(dict(lam=lam,macro=macro(y[va],vp.argmax(1)),mse=float(np.square(vp-np.eye(12)[y[va]]).mean())))
                ref=Ridge(alpha=lam*len(tr),solver='cholesky').fit((x[it]-mean)/scale,np.eye(12)[y[tr]],sample_weight=weights(y[tr],u[tr])*len(tr))
                err=float(abs(ref.predict((x[iv]-mean)/scale)-vp).max());assert err<1e-7;checks.append(dict(arm=arm,lam=lam,max_error=err))
            choice=min(range(len(candidates)),key=lambda j:(-candidates[j]['macro'],candidates[j]['mse'],-candidates[j]['lam']))
            sel[arm]={'selected':choice,'candidates':candidates,'dimension':x.shape[1]}
            for key,value in [('mean',mean),('scale',scale),('bias',bias),('coef',coefs[choice])]:models[arm+'_'+key]=value
            pred[arm]=score(x[ie],mean,scale,bias,coefs[choice])
        # Frozen selection complete; held scores cannot change any fitting decision.
        pred['cnn']=cnn[ie];assert digest(oldfolder/'predictions.npz')==old['files']['predictions.npz']
        with np.load(oldfolder/'predictions.npz') as z:np.testing.assert_array_equal(z['ids'],te);pred['tail_cnn']=z['tail']
        ss={k:macro(y[te],pred[k].argmax(1)) for k in scores}
        for k,v in ss.items():scores[k].append(v)
        dump(d/'split.json',{k:v.tolist() for k,v in parts.items()});dump(d/'selection.json',sel);np.savez_compressed(d/'models.npz',**models);np.savez_compressed(d/'predictions.npz',**pred)
        row=dict(held=held,scores=ss,checkpoint=str(oldfolder/'segment_24.pt'),checkpoint_sha256=sha,sklearn_checks=checks,files={p.name:digest(p) for p in sorted(d.iterdir())})
        dump(d/'result.json',row);rows.append(row);print(json.dumps({'completed':held,'seconds':time.perf_counter()-started}),flush=True)
    comparisons=decision(scores,c);summary=dict(state='complete',scores=scores,means={k:float(np.mean(v)) for k,v in scores.items()},comparisons=comparisons,gate_passed=all(r['passed'] for r in comparisons.values()),rows=rows,seconds=time.perf_counter()-started,encoder_training=False,selected_readouts=30,ridge_fits=90)
    dump(a.out/'SUMMARY.json',summary);dump(a.out/'FILES.json',{str(p.relative_to(a.out)):digest(p) for p in sorted(a.out.rglob('*')) if p.is_file()})
    print(json.dumps({k:v for k,v in summary.items() if k in ['state','means','gate_passed','seconds']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['data','prepared','models','protocol','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--cost-only',action='store_true');p.add_argument('--cost',type=Path);a=p.parse_args()
    with threadpool_limits(limits=1):main(a)
