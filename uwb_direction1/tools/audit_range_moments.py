"""Independent formulas for moments/scaling/metrics; checkpoint prediction replay."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--runs',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():raise ValueError('Preserve previous audit')
    summary=json.loads((a.runs/'SUMMARY.json').read_text())
    f=np.load(a.runs/'source_features.npz');ids=f['ids'];y=f['labels'];u=f['users'];seq=f['moments']
    assert sha(a.runs/'source_features.npz')==summary['features_sha256']
    all_y=np.load(a.data/'cache/y_clutterremoved.npy');all_u=np.load(a.data/'cache/subj_clutterremoved.npy')
    good=~(((all_u==2)&np.isin(all_y,[10,11]))|((all_u==6)&(all_y==3)))
    np.testing.assert_array_equal(ids,np.flatnonzero(good & np.isin(all_u,[3,4,5,6,7,8])))
    np.testing.assert_array_equal(y,all_y[ids]);np.testing.assert_array_equal(u,all_u[ids])
    raw=np.load(a.data/'cache/X_clutterremoved.npy',mmap_mode='r')
    checked=np.linspace(0,len(ids)-1,64,dtype=int)
    max_error=0.
    for k in checked:
        clip=raw[ids[k]]
        for view in range(3):
            for t in [0,11,44,68,89]:
                w=np.abs(clip[view,t].astype(np.float64))
                if w.sum()>0:
                    c=np.average(np.arange(189),weights=w)
                    s=np.sqrt(np.average((np.arange(189)-c)**2,weights=w))
                else:c=s=0.
                ref=np.array([c,s]);err=np.max(np.abs(ref-seq[k,view,:,t]))
                max_error=max(max_error,float(err))
                np.testing.assert_allclose(ref,seq[k,view,:,t],rtol=1e-12,atol=1e-12)
    scores={};artifacts=0
    for row in summary['rows']:
        arm=row['arm'];held=row['inner_held_user'];d=a.runs/arm/f'inner_user{held}'
        for name,h in row['sha256'].items():assert sha(d/name)==h;artifacts+=1
        split=json.loads((d/'split.json').read_text());tr=u!=held;te=u==held
        np.testing.assert_array_equal(split['train'],ids[tr]);np.testing.assert_array_equal(split['diagnostic'],ids[te])
        z=np.stack([seq[:,:,:,j:j+6].mean(-1) for j in range(0,90,6)],axis=-1)
        if arm=='static':z=np.tile(z.mean(-1,keepdims=True),(1,1,1,15))
        if arm=='ordered_relative':z[:,:,0,:]-=z[:,:,0,:].mean(-1,keepdims=True)
        if arm=='sorted_marginals':z=np.sort(z,axis=-1)
        x=z.reshape(len(ids),90)
        model=joblib.load(d/'model.joblib');scaler=model.steps[0][1]
        np.testing.assert_allclose(scaler.mean_,x[tr].mean(0),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(scaler.var_,x[tr].var(0),rtol=1e-12,atol=1e-12)
        pred=model.predict(x[te]);saved=np.load(d/'predictions.npz')
        np.testing.assert_array_equal(saved['ids'],ids[te]);np.testing.assert_array_equal(saved['y'],y[te])
        np.testing.assert_array_equal(saved['pred'],pred)
        score=balanced_accuracy_score(y[te],pred)
        assert abs(score-row['metrics']['macro_accuracy'])<1e-12
        scores.setdefault(arm,[]).append(score)
    assert len(summary['rows'])==24 and artifacts==72
    for arm,v in scores.items():assert abs(np.mean(v)-summary['means'][arm])<1e-12
    o=np.array(scores['ordered_absolute']);ds=o-np.array(scores['static']);du=o-np.array(scores['sorted_marginals'])
    gate=bool(o.mean()>=.85 and ds.mean()>=.05 and (ds>0).sum()>=4 and du.mean()>=.01 and (du>0).sum()>=4)
    assert gate==summary['continue_gate_passed']
    report={'state':'verified','runs':24,'artifact_sha256_checked':artifacts,'independent_moment_clip_checks':64,
        'independent_frame_view_checks':64*3*5,'max_moment_error_bins':max_error,
        'training_only_scaler_verified':True,'source_partition_verified':True,
        'saved_model_predictions_replayed':True,'independent_balanced_accuracy_recomputed':True,
        'continue_gate_recomputed':gate,'scope':'source-only diagnostic, no outer target inference',
        'summary_sha256':sha(a.runs/'SUMMARY.json'),'audit_script_sha256':sha(__file__)}
    a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
