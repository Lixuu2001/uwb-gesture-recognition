"""Prospective four-arm frozen-encoder source augmentation experiment."""
import argparse, json, time, platform
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from audit_restored import digest, dump
from temporal_reversal import Q, eligible, extract, augment
from run_energy_area import load_net
from second_order_core import fit_family, score, weights, macro
from radar_exchange import decision


def main(a):
    c = json.loads(a.protocol.read_text())
    assert c['label_map'] == Q.tolist() and not a.out.exists()
    a.out.mkdir(); dump(a.out/'PROTOCOL.json', c)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    paths = {k:Path(c['data'])/n for k,n in [('raw','X_clutterremoved.npy'),('labels','y_clutterremoved.npy'),('users','subj_clutterremoved.npy')]}
    paths['prepared'] = Path(c['prepared'])
    assert {k:digest(p) for k,p in paths.items()} == c['input_sha256']
    identity = dict(protocol_sha256=digest(a.protocol), python=platform.python_version(), torch=torch.__version__, code_sha256={n:digest(Path(__file__).with_name(n)) for n in ['temporal_reversal.py','run_temporal_reversal.py','circular_signal.py','run_energy_area.py','second_order_core.py','radar_exchange.py','audit_restored.py']})
    dump(a.out/'IDENTITY.json', identity)
    raw=np.load(paths['raw'],mmap_mode='r');cache=np.load(paths['prepared'],mmap_mode='r')
    old=Path(c['readout_reference']);feat=Path(c['feature_reference'])
    assert digest(old/'FILES.json')==c['readout_manifest_sha256'] and digest(feat/'FILES.json')==c['feature_manifest_sha256']
    rm=json.loads((old/'FILES.json').read_text());fm=json.loads((feat/'FILES.json').read_text())
    if a.cost_only:
        ids=np.array(json.loads((old/'held3/split.json').read_text())['train'][:16]);net,_,_=load_net(Path(c['models'])/'held3')
        t=time.perf_counter();extract(net,raw,cache,ids,c['batch_size']);secs=time.perf_counter()-t
        rng=np.random.default_rng(20261007);x=rng.normal(size=(8800,128));y=np.arange(len(x))%12;u=np.arange(len(x))%4
        t=time.perf_counter();fit_family(x,y,u,c['lambdas']);fitsecs=time.perf_counter()-t
        estimate=2*(6*4000/16*secs+18*fitsecs)
        out=dict(training16_seconds=secs,fit_seconds=fitsecs,estimated_seconds=estimate,passed=estimate<c['budget_seconds'],protocol_sha256=identity['protocol_sha256'],validation_or_diagnostic_used=False)
        dump(a.out/'COST.json',out);print(json.dumps(out));return
    cost=json.loads(a.cost.read_text());assert cost['passed'] and cost['protocol_sha256']==identity['protocol_sha256']
    y=np.load(paths['labels']);u=np.load(paths['users']);scores={k:[] for k in c['arms']};rows=[];t=time.perf_counter()
    for h in c['users']:
        d=a.out/f'held{h}';d.mkdir();od=old/f'held{h}';fd=feat/f'held{h}'
        for n in ['models.npz','predictions.npz','split.json','selection.json']:assert digest(od/n)==rm[f'held{h}/{n}']
        assert digest(fd/'features.npz')==fm[f'held{h}/features.npz']
        f=np.load(fd/'features.npz');m=np.load(od/'models.npz');split=json.loads((od/'split.json').read_text());sel=json.loads((od/'selection.json').read_text())['onehot']
        tr=np.array(split['train']);va=np.array(split['val']);te=np.array(split['diagnostic']);assert not(set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
        lookup={int(v):j for j,v in enumerate(f['ids'])};ix={r:np.array([lookup[int(v)] for v in ids]) for r,ids in [('train',tr),('val',va),('diagnostic',te)]}
        mask=eligible(y[tr],u[tr]);rid=tr[mask];net,sha,_=load_net(Path(c['models'])/f'held{h}');rev,naive=extract(net,raw,cache,rid,c['batch_size'])
        np.savez_compressed(d/'reverse_features.npz',ids=rid,reverse=rev,naive=naive)
        pred=dict(val_ids=va,diagnostic_ids=te);models={};selection={};checks=[]
        for role in ['val','diagnostic']:pred[role+'_ridge']=score(f['hidden'][ix[role]],m['mean'],m['scale'],m['bias'],m['coefs'][sel['selected']])
        for arm in c['arms'][1:]:
            x,yy,uu=augment(f['hidden'][ix['train']],y[tr],u[tr],rev,naive,arm);mean,scale,bias,coefs=fit_family(x,yy,uu,c['lambdas']);candidates=[]
            for lam,coef in zip(c['lambdas'],coefs):
                vp=score(f['hidden'][ix['val']],mean,scale,bias,coef);pred[arm+'_val_'+str(lam)]=vp
                candidates.append(dict(lam=lam,macro=macro(y[va],vp.argmax(1)),mse=float(np.mean((vp-np.eye(12)[y[va]])**2))))
                ref=Ridge(alpha=lam*len(x),solver='cholesky').fit((x-mean)/scale,np.eye(12)[yy],sample_weight=weights(yy,uu)*len(x));rr=ref.predict((f['hidden'][ix['val']]-mean)/scale)
                err=float(abs(rr-vp).max());assert err<1e-8;np.testing.assert_array_equal(rr.argmax(1),vp.argmax(1));checks.append(dict(arm=arm,lam=lam,max_error=err))
            j=min(range(3),key=lambda j:(-candidates[j]['macro'],candidates[j]['mse'],-candidates[j]['lam']));selection[arm]=dict(selected=j,candidates=candidates)
            for n,v in [('mean',mean),('scale',scale),('bias',bias),('coefs',np.array(coefs))]:models[arm+'_'+n]=v
            for role in ['val','diagnostic']:pred[role+'_'+arm]=score(f['hidden'][ix[role]],mean,scale,bias,coefs[j])
        np.savez_compressed(d/'models.npz',**models);np.savez_compressed(d/'predictions.npz',**pred);dump(d/'split.json',split);dump(d/'selection.json',selection);dump(d/'NUMERICAL_AUDIT.json',checks)
        # All held predictions saved before evaluation labels are opened.
        with np.load(od/'predictions.npz') as truth:
            np.testing.assert_array_equal(te,truth['ids']);np.testing.assert_allclose(pred['diagnostic_ridge'],truth['onehot']+.5,atol=1e-12,rtol=1e-12)
            ss={k:macro(truth['y'],pred['diagnostic_'+k].argmax(1)) for k in c['arms']};np.savez_compressed(d/'evaluation.npz',diagnostic_y=truth['y'],val_y=y[va])
        for k,v in ss.items():scores[k].append(v)
        row=dict(held=h,scores=ss,training_ids=len(tr),reversed_training_ids=len(rid),checkpoint_sha256=sha);rows.append(row);dump(d/'result.json',row);print(json.dumps(row),flush=True)
    comp=decision(scores,c);s=dict(state='complete',scores=scores,means={k:float(np.mean(v)) for k,v in scores.items()},comparisons=comp,gate_passed=all(v['passed'] for v in comp.values()),rows=rows,seconds=time.perf_counter()-t,readout_fits=54,encoder_training=False)
    dump(a.out/'SUMMARY.json',s);dump(a.out/'FILES.json',{str(p.relative_to(a.out)):digest(p) for p in sorted(a.out.rglob('*')) if p.is_file()});print(json.dumps({k:s[k] for k in ['means','gate_passed','seconds']}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['protocol','out']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--cost-only',action='store_true');p.add_argument('--cost',type=Path)
    with threadpool_limits(limits=1):main(p.parse_args())
