"""Post-evaluation replay; independent masks, weighted moments and ridge systems."""
import argparse,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from audit_restored import digest,dump


def audit(root):
    c=json.loads((root/'PROTOCOL.json').read_text());s=json.loads((root/'SUMMARY.json').read_text());manifest=json.loads((root/'FILES.json').read_text())
    for n,sha in manifest.items():assert digest(root/n)==sha
    identity=json.loads((root/'IDENTITY.json').read_text())
    for n,sha in identity['code_sha256'].items():assert digest(Path(__file__).with_name(n))==sha
    assert digest(root/'PROTOCOL.json')==identity['protocol_sha256']
    raw=Path('/workspace/uwb-restored-20260930/cache');y=np.load(raw/'y_clutterremoved.npy');u=np.load(raw/'subj_clutterremoved.npy')
    assert digest(raw/'y_clutterremoved.npy')=='d77b10f0fbda231246e71dfcf0b2f852863af48f80f159aed90f5f9b7c61a8bd'
    assert digest(raw/'subj_clutterremoved.npy')=='5ab32ab59e5b41c05d301b76949f159ddb1e9423a749deb760585a5f19cf9480'
    values={k:[] for k in c['arms']};maxerr=0.;models=0
    for h in c['users']:
        d=root/f'held{h}';p=np.load(d/'predictions.npz');m=np.load(d/'models.npz');masks=np.load(d/'subsets.npz');ev=np.load(d/'evaluation.npz');split=json.loads((d/'split.json').read_text());sel=json.loads((d/'selection.json').read_text());old=Path(c['readout_reference'])/f'held{h}';orig=json.loads((old/'selection.json').read_text())['onehot']
        assert sel['inherited_onehot']==orig and sel['new_selection'] is False
        assert sel['lambda_used']==orig['candidates'][orig['selected']]['lam']
        f=np.load(Path(c['feature_reference'])/f'held{h}/features.npz');lookup={int(v):i for i,v in enumerate(f['ids'])}
        def hidden(ids):return f['hidden'][[lookup[int(v)] for v in ids]]
        eligible=np.isin(u,[3,4,5,6,7,8])&~((u==6)&(y==3));v=h+1 if h<8 else 3
        roleids={'train':np.flatnonzero(eligible&(u!=h)&(u!=v)),'val':np.flatnonzero(eligible&(u==v)),'diagnostic':np.flatnonzero(eligible&(u==h))}
        for role,ids in roleids.items():np.testing.assert_array_equal(ids,split[role])
        tr=roleids['train']
        for deleted in sorted(set(u[tr])):
            expected=tr[u[tr]!=deleted];np.testing.assert_array_equal(expected,masks[f'domain_{deleted}'])
            removed=[]
            for k in range(12):
                pool=tr[y[tr]==k];n=int(((u[tr]==deleted)&(y[tr]==k)).sum());rng=np.random.default_rng(np.random.SeedSequence([c['seed'],h,int(deleted),k]));removed.extend(rng.choice(pool,n,replace=False))
            np.testing.assert_array_equal(np.setdiff1d(tr,removed),masks[f'row_{deleted}'])
            np.testing.assert_array_equal(np.bincount(y[expected],minlength=12),np.bincount(y[masks[f'row_{deleted}']],minlength=12))
        for name in masks.files:
            ii=masks[name];assert len(ii)==len(set(ii)) and set(ii)<=set(tr)
            users=np.unique(u[ii]);w=np.array([1/(len(users)*len(np.unique(y[ii][u[ii]==uu]))*np.sum((u[ii]==uu)&(y[ii]==yy))) for yy,uu in zip(y[ii],u[ii])]);x=hidden(ii);mu=np.sum(w[:,None]*x,axis=0);sd=np.sqrt(np.sum(w[:,None]*(x-mu)**2,axis=0));sd=np.where(sd>1e-12,sd,1);target=np.eye(12)[y[ii]];bias=np.sum(w[:,None]*target,axis=0);z=(x-mu)/sd;coef=np.linalg.solve(z.T@(w[:,None]*z)+sel['lambda_used']*np.eye(128),z.T@(w[:,None]*(target-bias)))
            for key,ref in [('mean',mu),('scale',sd),('bias',bias),('coef',coef)]:np.testing.assert_allclose(ref,m[name+'_'+key],atol=1e-9,rtol=1e-9)
            for role in ['val','diagnostic']:
                ref=(hidden(roleids[role])-mu)/sd@coef+bias;actual=p[role+'_'+name];maxerr=max(maxerr,float(abs(ref-actual).max()));np.testing.assert_allclose(ref,actual,atol=1e-9,rtol=1e-9);np.testing.assert_array_equal(ref.argmax(1),actual.argmax(1))
            models+=1
        for role in ['val','diagnostic']:
            np.testing.assert_array_equal(p[role+'_ids'],roleids[role]);np.testing.assert_array_equal(ev[role+'_y'],y[roleids[role]])
            for family in ['domain','row']:
                a=np.stack([p[role+'_'+n] for n in sorted(masks.files) if n.startswith(family+'_')]);np.testing.assert_array_equal(a.min(0),p[role+'_'+family+'_min']);np.testing.assert_array_equal(a.mean(0),p[role+'_'+family+'_mean'])
        for arm in values:
            truth=ev['diagnostic_y'];pred=p['diagnostic_'+arm].argmax(1);values[arm].append(float(np.mean([(pred[truth==k]==k).mean() for k in np.unique(truth)])))
    assert values==s['scores'];assert {k:float(np.mean(v)) for k,v in values.items()}==s['means']
    gates={}
    for arm,gain in c['gate']['minimum_mean_gain'].items():
        delta=np.array(values[c['primary']])-values[arm];gates[arm]=dict(mean_gain=float(delta.mean()),wins=int((delta>1e-12).sum()),minimum_gain=float(delta.min()),passed=bool(delta.mean()>=gain and (delta>1e-12).sum()>=c['gate']['minimum_wins'] and delta.min()>=-c['gate']['maximum_paired_decline']))
    assert gates==s['comparisons'] and all(g['passed'] for g in gates.values())==s['gate_passed']
    result=dict(state='passed',artifact_hashes=len(manifest),independent_refits=models,independent_role_predictions=models*2,max_error=maxerr,source_roles_ids_masks_counts_inherited_lambda_moments_scores_aggregation_truth_gate_verified=True,gate_passed=s['gate_passed']);dump(root/'AUDIT.json',result);print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    with threadpool_limits(limits=1):audit(a.root)
