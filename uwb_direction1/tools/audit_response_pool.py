"""Independent scalar response/bias/statistics spots and all sklearn ridge refits."""
import json,time,itertools
from pathlib import Path
import numpy as np
from scipy.signal import correlate
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from audit_restored import digest,dump


def response(x,view,signal,difference,dilation,kernel,channels):
    # Explicit range accumulation and scipy cross-correlation, not production convolution.
    z=sum(np.array(x[view,signal,:,int(r)],dtype=np.float64) for r in channels)/len(channels)
    if difference:z=z[1:]-z[:-1]
    w=np.zeros(8*dilation+1);w[::dilation]=kernel
    return correlate(z,w,mode='valid',method='direct')


def statistics(v,b):
    good=[j for j in range(len(v)) if v[j]>b];longest=run=0
    for x in v:
        run=run+1 if x>b else 0;longest=max(longest,run)
    return [len(good)/len(v),sum(float(v[j]-b) for j in good)/len(good) if good else 0.,sum(good)/len(good)/max(len(v)-1,1) if good else -1.,longest/len(v)]


def macro(y,p):return float(np.mean([np.mean(p[y==k]==k) for k in sorted(set(y))]))


def main():
    r=Path('/workspace/uwb-response-pool-source-20261004');p=Path('/workspace/uwb-temporal-publish-20261003/uwb_direction1');c=json.loads((r/'protocol.json').read_text());summary=json.loads((r/'SUMMARY.json').read_text());identity=json.loads((r/'IDENTITY.json').read_text());start=time.perf_counter()
    manifest=json.loads((r/'FILES.json').read_text())
    for n,sha in manifest.items():assert digest(r/n)==sha
    assert digest(p/'research/RESPONSE_POOL_PROTOCOL.json')==identity['protocol_sha256']
    for n,sha in identity['code_sha256'].items():assert digest(p/'tools'/n)==sha
    cache=np.load('/workspace/uwb-circular-prepared-20260930/signals.npy',mmap_mode='r');y=np.load('/workspace/uwb-restored-20260930/cache/y_clutterremoved.npy');u=np.load('/workspace/uwb-restored-20260930/cache/subj_clutterremoved.npy')
    expected_ids=np.flatnonzero((u>=3)&(u<=8)&~((u==6)&(y==3)));assert len(expected_ids)==7100
    # Independently reconstruct the fixed bank.
    rng=np.random.default_rng(c['bank_seed']);tri=np.array(list(itertools.combinations(range(9),3)))[rng.choice(84,14,replace=False)];kernels=-np.ones((14,9))
    for j,ix in enumerate(tri):kernels[j,ix]=2
    channels=np.stack([rng.choice(63,8,replace=False) for _ in range(14)])
    score_rows={k:[] for k in summary['scores']};bias_error=feature_error=fit_error=0.;bias_checks=feature_spots=refits=0
    for h in range(3,9):
        d=r/f'held{h}';split=json.loads((d/'split.json').read_text());val=3 if h==8 else h+1
        valid=(u>=3)&(u<=8)&~((u==6)&(y==3));expected={'train':np.flatnonzero(valid&(u!=h)&(u!=val)),'val':np.flatnonzero(valid&(u==val)),'diagnostic':np.flatnonzero(valid&(u==h))}
        for name,ii in expected.items():np.testing.assert_array_equal(ii,split[name])
        tr=expected['train'];va=expected['val'];te=expected['diagnostic'];lookup={int(i):j for j,i in enumerate(expected_ids)};it=[lookup[int(i)] for i in tr];iv=[lookup[int(i)] for i in va];ie=[lookup[int(i)] for i in te]
        with np.load(d/'bank.npz') as b:bank={k:b[k] for k in b.files}
        np.testing.assert_array_equal(kernels,bank['kernels']);np.testing.assert_array_equal(channels,bank['channels'])
        chosen=np.random.default_rng(c['bank_seed']).choice(tr,size=(36,14),replace=True);np.testing.assert_array_equal(chosen,bank['calibration_ids'])
        with np.load(d/'features.npz') as z:features={k:z[k].astype(np.float64) for k in c['arms']};np.testing.assert_array_equal(z['ids'],expected_ids)
        spots=np.concatenate([tr[:2],va[:2],te[:2]]);spotx=cache[spots];calx={int(i):cache[int(i)] for i in np.unique(chosen)}
        reconstructed={k:[] for k in c['arms']};block=0
        for view,signal,difference,dilation in itertools.product(range(3),[0,2],[False,True],[1,3,9]):
            for j in range(14):
                curve=response(calx[int(chosen[block,j])],view,signal,difference,dilation,kernels[j],channels[j]);bias=np.quantile(curve,[.2,.4,.5,.6,.8]);e=float(abs(bias-bank['biases'][block,j]).max());assert e<1e-9;bias_error=max(bias_error,e);bias_checks+=1
                per={k:[] for k in c['arms']}
                for n,sid in enumerate(spots):
                    v=response(spotx[n],view,signal,difference,dilation,kernels[j],channels[j]);b=bank['biases'][block,j];stat=statistics(v,b[2]);perm=np.random.default_rng(np.random.SeedSequence([c['shuffle_seed'],int(sid),len(v)])).permutation(len(v))
                    per['ppv_single'].append([stat[0]]);per['ppv_four'].append([sum(float(x)>bb for x in v)/len(v) for bb in b[[0,1,3,4]]]);per['multi_pool'].append(stat);per['temporal_null'].append(statistics(v[perm],b[2]))
                for name in per:reconstructed[name].append(np.array(per[name]));feature_spots+=len(spots)
            block+=1
        for name,blocks in reconstructed.items():
            z=np.concatenate(blocks,1).astype(np.float32).astype(np.float64);actual=features[name][[lookup[int(i)] for i in spots]]
            e=float(abs(z-actual).max());assert e<1e-6;feature_error=max(feature_error,e)
        with np.load(d/'models.npz') as z:models={k:z[k] for k in z.files}
        with np.load(d/'predictions.npz') as z:pred={k:z[k] for k in z.files}
        np.testing.assert_array_equal(pred['ids'],te);np.testing.assert_array_equal(pred['y'],y[te]);np.testing.assert_array_equal(pred['validation_ids'],va);np.testing.assert_array_equal(pred['validation_y'],y[va])
        selection=json.loads((d/'selection.json').read_text());w=np.zeros(len(tr))
        for user in sorted(set(u[tr])):
            cls=sorted(set(y[tr][u[tr]==user]))
            for label in cls:
                m=(u[tr]==user)&(y[tr]==label);w[m]=1/(4*len(cls)*sum(m))
        for arm,x in features.items():
            mean=np.average(x[it],axis=0,weights=w);scale=np.sqrt(np.average((x[it]-mean)**2,axis=0,weights=w));scale=np.where(scale>1e-12,scale,1.)
            np.testing.assert_allclose(mean,models[arm+'_mean'],atol=1e-10);np.testing.assert_allclose(scale,models[arm+'_scale'],atol=1e-10)
            candidates=[];fitted=[]
            for lam in c['lambdas']:
                model=Ridge(alpha=lam*len(tr),solver='cholesky').fit((x[it]-mean)/scale,np.eye(12)[y[tr]],sample_weight=w*len(tr));vp=model.predict((x[iv]-mean)/scale);refits+=1
                e=float(abs(vp-pred[f'{arm}_validation_{lam}']).max());assert e<1e-7;fit_error=max(fit_error,e)
                candidates.append(dict(lam=lam,macro=macro(y[va],vp.argmax(1)),mse=float(np.square(vp-np.eye(12)[y[va]]).mean())));fitted.append(model)
            j=min(range(3),key=lambda j:(-candidates[j]['macro'],candidates[j]['mse'],-candidates[j]['lam']));assert j==selection[arm]['selected']
            for a,b in zip(candidates,selection[arm]['candidates']):
                assert abs(a['macro']-b['macro'])<1e-12 and abs(a['mse']-b['mse'])<1e-10
            tp=fitted[j].predict((x[ie]-mean)/scale);e=float(abs(tp-pred[arm]).max());assert e<1e-7;fit_error=max(fit_error,e)
            np.testing.assert_allclose(fitted[j].coef_.T,models[arm+'_coef'],atol=1e-8);np.testing.assert_allclose(fitted[j].intercept_,models[arm+'_bias'],atol=1e-8)
        old=Path('/workspace/uwb-energy-area-source-20261004')/f'held{h}'/'predictions.npz';assert digest(old)==summary['rows'][h-3]['reference_predictions_sha256']
        with np.load(old) as z:
            for name in ['cnn_readout','tail_cnn']:np.testing.assert_array_equal(pred[name],z[name])
        for name in score_rows:score_rows[name].append(macro(y[te],pred[name].argmax(1)))
        print(json.dumps({'audited_held':h}),flush=True)
    for name,row in score_rows.items():np.testing.assert_allclose(row,summary['scores'][name],atol=1e-14)
    checks={}
    for name,threshold in c['gate']['minimum_mean_gain'].items():
        delta=np.array(score_rows['multi_pool'])-score_rows[name];passed=bool(delta.mean()>=threshold and sum(delta>1e-12)>=c['gate']['minimum_wins'] and delta.min()>=-c['gate']['maximum_paired_decline']);assert passed==summary['comparisons'][name]['passed'];checks[name]=passed
    assert all(checks.values())==summary['gate_passed']
    result=dict(state='passed',artifact_hashes=len(manifest),calibration_curves=bias_checks,scalar_feature_statistic_set_checks=feature_spots,feature_samples_per_fold=6,all_features_independently_recomputed=False,sklearn_refits=refits,max_bias_error=bias_error,max_feature_error=feature_error,max_prediction_error=fit_error,roles_and_selection_verified=True,gate_reproduced=True,seconds=time.perf_counter()-start)
    dump(r/'AUDIT.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
