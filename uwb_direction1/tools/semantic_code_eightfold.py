"""Fixed known output-code ridge adaptation; no novel representation claim."""
import argparse,json,time,platform
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from audit_restored import digest,dump
from semantic_code_eightfold_prepare import roles as source_roles
from second_order_core import fit_family,score,macro


def codes(seed):
    a=np.zeros((12,8))
    for i,cols in enumerate([[0],[1],[2],[3],[0,2],[0,3],[1,2],[1,3],[4],[5],[6],[7]]):a[i,cols]=1
    a/=np.linalg.norm(a,axis=1,keepdims=True)
    permutation=np.random.default_rng(seed).permutation(12)
    q,_=np.linalg.qr(np.random.default_rng(seed+1).normal(size=(20,12)))
    return {'onehot':np.eye(12),'semantic':np.concatenate([np.eye(12),a],1)/np.sqrt(2),
            'permuted':np.concatenate([np.eye(12),a[permutation]],1)/np.sqrt(2),'orthogonal':q.T}


def decode(p,c):return p@c@c.T-.5*np.square(c).sum(1)


def choice(candidates):return min(range(len(candidates)),key=lambda j:(-candidates[j]['macro'],candidates[j]['mse'],-candidates[j]['lam']))


def main(a):
    cfg=json.loads(a.protocol.read_text());assert not a.out.exists();a.out.mkdir();assert digest(a.source/'SUMMARY.json')==cfg['source_gate_summary_sha256'];assert digest(a.source/'AUDIT.json')==cfg['source_gate_audit_sha256'];assert json.loads((a.source/'SUMMARY.json').read_text())['gate_passed']
    identity=dict(protocol_sha256=digest(a.protocol),python=platform.python_version(),numpy=np.__version__,code_sha256={n:digest(Path(__file__).with_name(n)) for n in ['semantic_code_eightfold.py','semantic_code_eightfold_prepare.py','second_order_core.py']})
    dump(a.out/'IDENTITY.json',identity);dump(a.out/'PROTOCOL.json',cfg)
    if a.cost_only:
        rng=np.random.default_rng(21);x=rng.normal(size=(4800,128));v=rng.normal(size=(1200,128));t=time.perf_counter()
        mean,scale,bias,bs=fit_family(x,np.arange(4800)%12,np.arange(4800)%4,cfg['lambdas'])
        for b in bs:score(v,mean,scale,bias,b)
        elapsed=time.perf_counter()-t;estimate=elapsed*18*3
        out=dict(synthetic_seconds=elapsed,estimated_seconds=estimate,budget=cfg['budget_seconds'],passed=estimate<cfg['budget_seconds'],real_data_or_scores_read=False)
        dump(a.out/'COST.json',out);print(json.dumps(out));return
    assert json.loads(a.cost.read_text())['passed'];assert json.loads(a.cost.read_text())['protocol_sha256']==identity['protocol_sha256']
    assert json.loads((a.reference/'SOURCE_IDENTITY.json').read_text())['checkpoint_files']==cfg['checkpoint_files'];ref=json.loads((a.reference/'FILES.json').read_text())
    yp=a.data/'cache/y_clutterremoved.npy';up=a.data/'cache/subj_clutterremoved.npy'
    assert digest(yp)==cfg['input_sha256']['labels'] and digest(up)==cfg['input_sha256']['users']
    y=np.load(yp);u=np.load(up);ids=np.flatnonzero(~(((u==2)&np.isin(y,[10,11]))|((u==6)&(y==3))));assert len(ids)==9300
    lookup=np.full(len(y),-1);lookup[ids]=np.arange(len(ids));cc=codes(cfg['seed']);np.savez_compressed(a.out/'codes.npz',**cc)
    start=time.perf_counter();scores={k:[] for k in cfg['arms']+['cnn']};rows=[]
    for h in cfg['source_users']:
        old=a.reference/f'held{h}';d=a.out/f'held{h}';d.mkdir();parts=source_roles(y,u,h)
        for n in ['features.npz','split.json','predictions.npz']:assert digest(old/n)==ref[f'held{h}/{n}']
        split=json.loads((old/'split.json').read_text())
        for k,v in parts.items():np.testing.assert_array_equal(v,split[k])
        with np.load(old/'features.npz') as z:x=z['hidden'];np.testing.assert_array_equal(ids,z['ids'])
        tr,va,te=[parts[k] for k in ['train','val','diagnostic']];it,iv,ie=[lookup[v] for v in [tr,va,te]]
        mean,scale,bias,bs=fit_family(x[it],y[tr],u[tr],cfg['lambdas']);vps=[score(x[iv],mean,scale,bias,b) for b in bs]
        selection={};pred={'ids':te,'y':y[te],'validation_ids':va,'validation_y':y[va]};model={'mean':mean,'scale':scale,'bias':bias,'coefs':np.array(bs)}
        for arm,c in cc.items():
            candidates=[]
            for j,(lam,vp) in enumerate(zip(cfg['lambdas'],vps)):
                z=decode(vp,c);pred[f'{arm}_validation_{j}']=z
                candidates.append(dict(lam=lam,macro=macro(y[va],z.argmax(1)),mse=float(np.square(vp-np.eye(12)[y[va]]).mean())))
            selection[arm]=dict(selected=choice(candidates),candidates=candidates)
        dump(d/'selection.json',selection) # Complete all choices before diagnostic forward.
        for arm,c in cc.items():pred[arm]=decode(score(x[ie],mean,scale,bias,bs[selection[arm]['selected']]),c)
        with np.load(old/'predictions.npz') as z:
            np.testing.assert_array_equal(z['ids'],te);pred['cnn']=z['cnn']
        np.testing.assert_array_equal(pred['onehot'].argmax(1),pred['orthogonal'].argmax(1));assert selection['onehot']['selected']==selection['orthogonal']['selected']
        ss={k:macro(y[te],pred[k].argmax(1)) for k in scores}
        for k,v in ss.items():scores[k].append(v)
        dump(d/'split.json',{k:v.tolist() for k,v in parts.items()});np.savez_compressed(d/'models.npz',**model);np.savez_compressed(d/'predictions.npz',**pred)
        row=dict(held=h,scores=ss);dump(d/'result.json',row);rows.append(row);print(json.dumps(row),flush=True)
    comparisons={}
    for k,threshold in cfg['gate']['minimum_mean_gain'].items():
        delta=np.array(scores['semantic'])-scores[k];comparisons[k]=dict(mean_gain=float(delta.mean()),wins=int((delta>1e-12).sum()),minimum_gain=float(delta.min()),passed=bool(delta.mean()>=threshold and (delta>1e-12).sum()>=cfg['gate']['minimum_wins'] and delta.min()>=-cfg['gate']['maximum_paired_decline']))
    summary=dict(state='complete',scores=scores,means={k:float(np.mean(v)) for k,v in scores.items()},comparisons=comparisons,gate_passed=all(v['passed'] for v in comparisons.values()),seconds=time.perf_counter()-start,rows=rows,encoder_training=False)
    dump(a.out/'SUMMARY.json',summary);dump(a.out/'FILES.json',{str(p.relative_to(a.out)):digest(p) for p in sorted(a.out.rglob('*')) if p.is_file()});print(json.dumps({k:summary[k] for k in ['means','gate_passed','seconds']}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['protocol','out','reference','data','source']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--cost-only',action='store_true');p.add_argument('--cost',type=Path);a=p.parse_args()
    with threadpool_limits(limits=1):main(a)
