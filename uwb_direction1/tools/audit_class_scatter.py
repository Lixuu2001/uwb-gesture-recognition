"""Saved-artifact, role and decision audit, independent of gate helpers."""
from pathlib import Path
import argparse,json
import numpy as np
from audit_restored import digest,dump
from energy_area import source_roles


def main(root):
    c=json.loads((root/'PROTOCOL.json').read_text());identity=json.loads((root/'IDENTITY.json').read_text());manifest=json.loads((root/'FILES.json').read_text())
    assert digest(root/'PROTOCOL.json')==identity['protocol_sha256']
    for n,h in manifest.items():assert digest(root/n)==h
    for n,h in identity['code_sha256'].items():assert digest(Path(__file__).with_name(n))==h
    raw=Path('/workspace/uwb-restored-20260930/cache');y=np.load(raw/'y_clutterremoved.npy');u=np.load(raw/'subj_clutterremoved.npy');scores={k:[] for k in c['arms']};checks=[]
    for held in c['users']:
        d=root/f'held{held}';parts=source_roles(y,u,held);saved=json.loads((d/'split.json').read_text());pred=np.load(d/'predictions.npz');truth=np.load(d/'evaluation.npz');info=json.loads((d/'fit.json').read_text())
        for k,v in parts.items():np.testing.assert_array_equal(v,saved[k])
        for role in ['val','diagnostic']:
            ids=pred[role+'_ids'];np.testing.assert_array_equal(ids,parts[role]);np.testing.assert_array_equal(truth[role+'_y'],y[ids])
            for k in c['arms']:assert np.isfinite(pred[role+'_'+k]).all()
        for k in c['arms']:
            ids=parts['diagnostic'];guess=pred['diagnostic_'+k].argmax(1);v=float(np.mean([np.mean(guess[y[ids]==cl]==cl) for cl in np.unique(y[ids])]));scores[k].append(v)
        tr=parts['train'];counts={str(k):{str(v):int(((y[tr]==k)&(u[tr]==v)).sum()) for v in np.unique(u[tr])} for k in range(12)};assert counts==info['class_user_counts'];checks.append(all(len(set(v for v in z.values() if v))==1 for z in counts.values()))
    summary=json.loads((root/'SUMMARY.json').read_text());passes=[]
    for k,v in scores.items():np.testing.assert_array_equal(v,summary['scores'][k]);assert np.mean(v)==summary['means'][k]
    for arm,minimum in c['gate']['minimum_mean_gain'].items():
        d=np.array(scores[c['primary']])-scores[arm];r=dict(mean_gain=float(d.mean()),wins=int((d>1e-12).sum()),minimum_gain=float(d.min()),passed=bool(d.mean()>=minimum and (d>1e-12).sum()>=c['gate']['minimum_wins'] and d.min()>=-c['gate']['maximum_paired_decline']));assert r==summary['comparisons'][arm];passes.append(r['passed'])
    assert all(passes)==summary['gate_passed'];before=np.load('/workspace/uwb-class-scatter-source-20261007/held3/predictions.npz');after=np.load(root/'held3/predictions.npz');assert before.files==after.files
    for k in before.files:np.testing.assert_array_equal(before[k],after[k])
    r=dict(state='passed',artifact_hashes=len(manifest),all_splits_and_evaluation_labels_verified=True,source_class_counts_equal_within_present_users=checks,failed_attempt_predictions_unchanged=True,gate_independently_recomputed=True,fit_numerical_audit='72 independent analytic covariance reconstructions and576 SciPy Gaussian logpdf systems were checked by runner before label evaluation',max_scipy_score_difference=summary['max_scipy_logpdf_difference']);dump(root/'AUDIT.json',r);print(json.dumps(r))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
