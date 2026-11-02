"""Fixed source-only range-moment diagnostic, no outer target signal access."""
import argparse
import json
from pathlib import Path
import time
import joblib
import numpy as np
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from audit_restored import digest, dump, expected_parts, metrics


def moments(raw):
    """Return [N,3,2,90] centroid/spread of abs RF; coordinates are bins."""
    a = np.abs(np.asarray(raw, dtype=np.float64))
    if a.ndim != 4 or a.shape[1:] != (3,90,189):
        raise ValueError('Expected [N,3,90,189]')
    mass = a.sum(-1, keepdims=True)
    p = np.divide(a, mass, out=np.zeros_like(a), where=mass > 0)
    bins = np.arange(189, dtype=np.float64)
    center = (p*bins).sum(-1)
    spread = np.sqrt((p*(bins-center[...,None])**2).sum(-1))
    return np.stack([center,spread],axis=2)


def features(seq, arm):
    z = seq.reshape(len(seq),3,2,15,6).mean(-1)
    if arm == 'static':
        z = np.repeat(z.mean(-1,keepdims=True),15,axis=-1)
    elif arm == 'ordered_relative':
        z = z.copy()
        z[:,:,0] -= z[:,:,0].mean(-1,keepdims=True)
    elif arm == 'sorted_marginals':
        z = np.sort(z,axis=-1)
    elif arm != 'ordered_absolute':
        raise ValueError(arm)
    return z.reshape(len(z),90)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise ValueError('Preserve existing output; use new directory')
    config=json.loads(args.protocol.read_text())
    y=np.load(args.data/'cache/y_clutterremoved.npy')
    u=np.load(args.data/'cache/subj_clutterremoved.npy')
    ids=expected_parts(y,u,1)['train']
    assert np.array_equal(np.unique(u[ids]),config['source_users'])
    assert not np.isin(u[ids],config['excluded_signal_users']).any()
    raw=np.load(args.data/'cache/X_clutterremoved.npy',mmap_mode='r')
    args.out.mkdir(parents=True)
    dump(args.out/'PROTOCOL.json',config)
    start=time.perf_counter()
    seq=np.concatenate([moments(raw[ids[i:i+32]]) for i in range(0,len(ids),32)])
    assert np.isfinite(seq).all()
    np.savez_compressed(args.out/'source_features.npz',ids=ids,labels=y[ids],users=u[ids],moments=seq)
    source_y,source_u=y[ids],u[ids]
    rows=[]
    for arm in config['arms']:
        x=features(seq,arm)
        for held in config['source_users']:
            train=np.flatnonzero(source_u!=held);test=np.flatnonzero(source_u==held)
            assert not np.intersect1d(ids[train],ids[test]).size
            model=make_pipeline(StandardScaler(),SVC(C=10,gamma=1/90,class_weight='balanced',cache_size=256))
            tick=time.perf_counter();model.fit(x[train],source_y[train])
            pred=model.predict(x[test]);score=metrics(source_y[test],pred)
            d=args.out/arm/f'inner_user{held}';d.mkdir(parents=True)
            joblib.dump(model,d/'model.joblib')
            np.savez_compressed(d/'predictions.npz',ids=ids[test],y=source_y[test],pred=pred)
            dump(d/'split.json',{'train':ids[train].tolist(),'diagnostic':ids[test].tolist()})
            row={'arm':arm,'inner_held_user':held,'metrics':score,'seconds':time.perf_counter()-tick,
                 'sha256':{p.name:digest(p) for p in d.iterdir()}}
            dump(d/'result.json',row);rows.append(row)
            print(json.dumps({'arm':arm,'inner_held_user':held,'macro_accuracy':score['macro_accuracy']}),flush=True)
    scores={a:[r['metrics']['macro_accuracy'] for r in rows if r['arm']==a] for a in config['arms']}
    ordered=np.array(scores['ordered_absolute'])
    static_delta=ordered-np.array(scores['static']);sorted_delta=ordered-np.array(scores['sorted_marginals'])
    passed=bool(ordered.mean()>=.85 and static_delta.mean()>=.05 and (static_delta>0).sum()>=4
                and sorted_delta.mean()>=.01 and (sorted_delta>0).sum()>=4)
    dump(args.out/'SUMMARY.json',{'scope':config['scope'],'source_samples':len(ids),'source_users':config['source_users'],
        'outer_target_signals_read':False,'scores':scores,'means':{a:float(np.mean(v)) for a,v in scores.items()},
        'ordered_minus_static':static_delta.tolist(),'ordered_minus_sorted':sorted_delta.tolist(),
        'continue_gate_passed':passed,'seconds':time.perf_counter()-start,
        'script_sha256':digest(__file__),'protocol_sha256':digest(args.protocol),
        'features_sha256':digest(args.out/'source_features.npz'),'numpy':np.__version__,'sklearn':sklearn.__version__,
        'rows':rows})


if __name__=='__main__':main()
