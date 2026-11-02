"""Independent neighbors/dense-system replay. No production graph import."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.linalg import solve
from threadpoolctl import threadpool_limits
from audit_restored import digest,dump

def macro(y,p):return float(np.mean([np.mean(p[y==k]==k) for k in np.unique(y)]))

def main(a):
    c=json.loads((a.run/'PROTOCOL.json').read_text());manifest=json.loads((a.run/'FILES.json').read_text())
    for n,h in manifest.items():assert digest(a.run/n)==h
    old=Path(c['readout_reference']);feat=Path(c['feature_reference']);assert digest(old/'FILES.json')==c['readout_manifest_sha256'];assert digest(feat/'FILES.json')==c['feature_manifest_sha256'];rm=json.loads((old/'FILES.json').read_text());fm=json.loads((feat/'FILES.json').read_text())
    all_scores={arm:[] for arm in c['arms']};maxerror=0;solves=0;graph_entries=0
    for h in c['users']:
        d=a.run/f'held{h}';od=old/f'held{h}';fd=feat/f'held{h}'
        for n in ['models.npz','predictions.npz','split.json','selection.json']:assert digest(od/n)==rm[f'held{h}/{n}']
        assert digest(fd/'features.npz')==fm[f'held{h}/features.npz'];splits=json.loads((d/'split.json').read_text());assert splits==json.loads((od/'split.json').read_text());sel=json.loads((d/'selection.json').read_text());assert not sel['new_selection'] and sel['inherited_onehot']==json.loads((od/'selection.json').read_text())['onehot']
        checkpoint=np.load(od/'models.npz');f=np.load(fd/'features.npz');lookup={int(v):i for i,v in enumerate(f['ids'])};saved=np.load(d/'predictions.npz');truth=np.load(od/'predictions.npz');evaluation=np.load(d/'evaluation.npz')
        for role,ik,yk in [('val','validation_ids','validation_y'),('diagnostic','ids','y')]:
            ii=np.array(splits[role]);np.testing.assert_array_equal(ii,saved[role+'_ids']);np.testing.assert_array_equal(ii,truth[ik]);np.testing.assert_array_equal(evaluation[role+'_y'],truth[yk]);x=(f['hidden'][[lookup[int(v)] for v in ii]]-checkpoint['mean'])/checkpoint['scale'];j=sel['inherited_onehot']['selected'];unary=x@checkpoint['coefs'][j]+checkpoint['bias'];p=np.maximum(unary,1e-12);p/=p.sum(1)[:,None];np.testing.assert_allclose(p,saved[role+'_ridge'],atol=1e-12)
            order=np.argsort(ii);inv=np.argsort(order);z=x[order];norm=np.sqrt(np.sum(z*z,axis=1));z=z/np.where(norm>0,norm,1)[:,None];n=len(z);adj=np.zeros((n,n))
            for row in range(n):
                dist=np.sum((z-z[row])**2,axis=1);dist[row]=np.inf;nn=np.lexsort((ii[order],dist))[:min(c['k'],n-1)];adj[row,nn]=1
            mutual=adj*adj.T;union=(adj+adj.T)/2;perm=np.random.default_rng(c['seed']).permutation(n);ww={'mutual':mutual,'union':union,'permuted':mutual[perm][:,perm]}
            for arm,v in ww.items():
                w=v[inv][:,inv];stored=sparse.load_npz(d/f'{role}_{arm}_graph.npz').toarray();np.testing.assert_array_equal(stored,w);graph_entries+=int(np.count_nonzero(w));assert np.all(w>=0) and np.all(np.diag(w)==0);np.testing.assert_array_equal(w,w.T)
                system=np.eye(n)+c['mu']*(np.diag(w.sum(1))-w);out=solve(system,p,assume_a='pos');solves+=1;err=float(abs(out-saved[role+'_'+arm]).max());maxerror=max(maxerror,err);np.testing.assert_allclose(out,saved[role+'_'+arm],atol=1e-10,rtol=1e-10);np.testing.assert_array_equal(out.argmax(1),saved[role+'_'+arm].argmax(1));np.testing.assert_allclose(out.sum(1),1,atol=1e-10);assert out.min()>-1e-12 and abs(system@out-p).max()<1e-10
            if role=='diagnostic':
                for arm in c['arms']:all_scores[arm].append(macro(truth['y'],saved[role+'_'+arm].argmax(1)))
    summary=json.loads((a.run/'SUMMARY.json').read_text());passed=[]
    for arm in all_scores:np.testing.assert_allclose(summary['scores'][arm],all_scores[arm],atol=1e-14);np.testing.assert_allclose(summary['means'][arm],np.mean(all_scores[arm]),atol=1e-14)
    for arm,gain in c['gate']['minimum_mean_gain'].items():
        delta=np.array(all_scores['mutual'])-all_scores[arm];v=bool(delta.mean()>=gain and (delta>1e-12).sum()>=c['gate']['minimum_wins'] and delta.min()>=-c['gate']['maximum_paired_decline']);assert summary['comparisons'][arm]['passed']==v;passed.append(v)
    assert summary['gate_passed']==all(passed)
    out=dict(state='passed',dense_replayed_systems=solves,replayed_nonzero_graph_entries=graph_entries,maximum_prediction_absolute_error=maxerror,artifact_sha_verified=len(manifest),all_predictions_and_graphs_verified=True,gate_recomputed=all(passed),summary_sha256=digest(a.run/'SUMMARY.json'));dump(a.run/'AUDIT.json',out);print(json.dumps(out))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):main(a)
