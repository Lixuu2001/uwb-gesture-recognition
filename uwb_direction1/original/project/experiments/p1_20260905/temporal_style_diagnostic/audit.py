"""CPU audit: recompute diagnostics from saved logits and correspondence maps.

Does not import diagnostic transforms or their summary implementations.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,f1_score

HERE=Path(__file__).resolve().parent; BASE=HERE.parent; ROOT=HERE/'runs'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def metric(z,y):
    p=z.argmax(1); q=z.astype(float); q-=q.max(1,keepdims=True)
    return {'accuracy':float(accuracy_score(y,p)),
            'macro_f1':float(f1_score(y,p,labels=np.unique(y),average='macro',zero_division=0)),
            'ce':float(np.mean(np.log(np.exp(q).sum(1))-q[np.arange(len(y)),y]))}

def main():
    manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    for name,h in manifest['source_hashes'].items(): assert sha(BASE/name)==h
    y=np.load(BASE/'prepared/y.npy'); s=np.load(BASE/'prepared/subj.npy')
    good=~(((s==2)&np.isin(y,[10,11]))|((s==6)&(y==3)))
    folds=[]; source_ids=[]; val_ids=[]
    for fold in range(1,9):
        d=ROOT/f'fold{fold}'; completed=json.loads((d/'complete.json').read_text(encoding='utf-8'))
        train=np.flatnonzero(good&~np.isin(s,[fold,fold%8+1]))
        val=np.flatnonzero(good&(s==fold%8+1))
        p=np.load(d/'validation.npz'); ids=p['val_ids']; val_ids.extend(ids.tolist())
        assert np.array_equal(ids,val) and np.array_equal(p['labels'],y[val])
        assert np.isin(p['donor_ids'],train).all()
        checkpoint=BASE/'cross_user_meta/runs_s2026'/f'fold{fold}/warmup/right.pt'
        assert sha(checkpoint)==completed['model_hash']
        old=np.load(checkpoint.parent/'predictions.npz')
        assert np.array_equal(p['clean_logits'],old['val_logits'])
        base=p['clean_logits'].argmax(1); perf={}; layer={}
        for name in p.files:
            if name.endswith('_logits'):
                z=p[name]; assert z.shape==(len(val),12) and np.isfinite(z).all()
                m=metric(z,y[val]); pred=z.argmax(1)
                m.update(flip_rate=float(np.mean(pred!=base)),
                         correct_to_wrong=float(np.mean((base==y[val])&(pred!=y[val]))),
                         wrong_to_correct=float(np.mean((base!=y[val])&(pred==y[val]))))
                perf[name[:-7]]=m
            elif name.startswith('style_'):
                assert p[name].shape==(len(val),) and np.isfinite(p[name]).all()
                layer[name]=float(p[name].mean())
        a=np.load(d/'alignment_inputs.npz'); ids=a['ids']; source_ids.extend(ids.tolist())
        assert np.isin(ids,train).all() and len(np.unique(ids))==len(ids)
        bad=a['mismatch_partner']; other=a['cross_user_partner']
        assert np.isin(bad,ids).all() and np.isin(other,ids).all()
        assert np.all(bad!=ids) and np.array_equal(y[bad],y[ids]) and np.array_equal(s[bad],s[ids])
        assert np.array_equal(y[other],y[ids]) and np.all(s[other]!=s[ids])
        rows=json.loads((d/'alignment_rows.json').read_text(encoding='utf-8'))
        assert len(rows['synthetic'])==len(ids)*2*5
        assert len(rows['cross_user_equivariance'])==len(ids)*2*3
        assert a['synthetic_maps'].shape==(len(rows['synthetic']),90)
        for row,path in zip(rows['synthetic'],a['synthetic_maps']):
            t=np.arange(90); g=t+row['amplitude']*np.sin(2*np.pi*t/89)
            err=np.abs(np.interp(path,t,g)-t)[5:85]
            assert abs(err.mean()-row['mae_frames'])<1e-10
            assert abs(np.mean(err<=2)-row['within2'])<1e-10
            assert row['id'] in ids and row['user']==int(s[row['id']]) and row['class']==int(y[row['id']])
        summary={}
        for kind,rr in rows.items():
            summary[kind]={}
            for method in sorted({r['method'] for r in rr}):
                selected=[r for r in rr if r['method']==method]
                summary[kind][method]={'mae_frames':float(np.mean([r['mae_frames'] for r in selected]))}
                if kind=='synthetic': summary[kind][method]['within2']=float(np.mean([r['within2'] for r in selected]))
                else: summary[kind][method]['path_deviation_frames']=float(np.mean([r['path_deviation_frames'] for r in selected]))
        folds.append({'fold':fold,'validation_user':fold%8+1,'validation_samples':len(val),
                      'source_samples':len(ids),'performance':perf,'layer_change':layer,'alignment':summary,
                      'seconds':completed['seconds']})
    assert len(val_ids)==len(set(val_ids))==int(good.sum())
    avgperf={name:{key:float(np.mean([f['performance'][name][key] for f in folds]))
                   for key in folds[0]['performance'][name]} for name in folds[0]['performance']}
    for name,m in avgperf.items():
        m['delta_accuracy_pp']=100*(m['accuracy']-avgperf['clean']['accuracy'])
        m['per_user_delta_pp']=[100*(f['performance'][name]['accuracy']-f['performance']['clean']['accuracy']) for f in folds]
    avgalign={kind:{method:{key:float(np.mean([f['alignment'][kind][method][key] for f in folds]))
                           for key in folds[0]['alignment'][kind][method]}
                    for method in folds[0]['alignment'][kind]} for kind in folds[0]['alignment']}
    avgstyle={key:float(np.mean([f['layer_change'][key] for f in folds])) for key in folds[0]['layer_change']}
    result={'status':'verified','folds':folds,'validation_samples':len(val_ids),
            'source_sample_exposures':len(source_ids),'unique_source_samples':len(set(source_ids)),
            'validation_logits_replayed':len(val_ids)*len(avgperf),
            'equal_user_performance':avgperf,'equal_fold_alignment':avgalign,'equal_user_layer_change':avgstyle,
            'seconds':sum(f['seconds'] for f in folds),'source_hashes_verified':len(manifest['source_hashes']),
            'config':manifest['config'],
            'limits':['Frame synchronization assumed by user; not verified.',
                      'Validation results are development diagnostics, not outer-test performance.',
                      'Warping prepared amplitude/phase maps does not guarantee physically valid or label-preserving motion.',
                      'Synthetic correspondence is known; cross-person semantic correspondence is not labeled.',
                      'Training sample exposures overlap across folds; do not treat fold alignment means as independent trials.',
                      'Cross-user equivariance errors summarized from stored rows; synthetic errors independently recalculated from maps.']}
    (ROOT/'AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['folds','config']},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
