"""Full independent image/tensor/descriptor replay and all sklearn refits."""
import json,time
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from audit_restored import digest,dump
from audit_orientation_screen import independent_image
from check_orientation_screen import reference_tensor
from check_orientation_features import scalar_hist


def independent_descriptor(tensors,sid,seed):
    result={k:[] for k in ['all','coherent','sign_null','cell_null']}
    for view,z in enumerate(tensors):
        matrix=np.empty((126,2,2));matrix[:,0,0]=z[:,0];matrix[:,0,1]=matrix[:,1,0]=z[:,1];matrix[:,1,1]=z[:,2];vals,vectors=np.linalg.eigh(matrix);trace=z[:,0]+z[:,2];gap=vals[:,1]-vals[:,0]
        coh=np.divide(gap,trace,out=np.zeros(126),where=trace>1e-12);v=vectors[:,:,-1];ob=2*abs(v[:,0]*v[:,1]);angle=(np.arctan2(v[:,1],v[:,0])+np.pi/2)%np.pi-np.pi/2
        eligible=(trace>=np.quantile(trace,.75))&(trace>1e-12);usable=eligible&(coh>=.6)&(ob>=.3);den=trace*eligible
        full=scalar_hist(angle,trace*coh*eligible,den);primary=scalar_hist(angle,trace*coh*usable,den)
        signs=np.random.default_rng(np.random.SeedSequence([seed,int(sid),view,0])).choice([-1,1],size=126);sn=scalar_hist(angle*signs,trace*coh*usable,den);perm=np.random.default_rng(np.random.SeedSequence([seed,int(sid),view,1])).permutation(6)
        for name,z in [('all',full),('coherent',primary),('sign_null',sn),('cell_null',primary[perm])]:result[name].append(z.reshape(-1))
    return {k:np.concatenate(v) for k,v in result.items()}


def macro(y,p):return float(np.mean([np.mean(p[y==c]==c) for c in np.unique(y)]))


def main():
    R=Path('/workspace/uwb-orientation-readout-20261004');P=Path('/workspace/uwb-temporal-publish-20261003/uwb_direction1');B=Path('/workspace/uwb-energy-area-source-20261004');start=time.perf_counter();c=json.loads((R/'PROTOCOL.json').read_text());s=json.loads((R/'SUMMARY.json').read_text());ident=json.loads((R/'IDENTITY.json').read_text());manifest=json.loads((R/'FILES.json').read_text())
    for n,sha in manifest.items():assert digest(R/n)==sha
    assert digest(P/'research/ORIENTATION_READOUT_PROTOCOL.json')==ident['protocol_sha256']
    for n,sha in ident['code_sha256'].items():assert digest(P/'tools'/n)==sha
    raw=np.load('/workspace/uwb-restored-20260930/cache/X_clutterremoved.npy',mmap_mode='r');u=np.load('/workspace/uwb-restored-20260930/cache/subj_clutterremoved.npy');y=np.load('/workspace/uwb-restored-20260930/cache/y_clutterremoved.npy');ids=np.flatnonzero((u>=3)&(u<=8)&~((u==6)&(y==3)));assert len(ids)==7100
    with np.load(R/'descriptors.npz') as z:tensors=z['tensors'];desc={k:z[k] for k in c['arms']};np.testing.assert_array_equal(ids,z['ids'])
    tensor_error=descriptor_error=0.
    for j,sid in enumerate(ids):
        im=independent_image(raw[int(sid)]);ref=np.stack([reference_tensor(v)[10:80:5,10:53:5].reshape(126,3) for v in im]);e=float(abs(ref-tensors[j]).max());assert e<1e-12;tensor_error=max(e,tensor_error);features=independent_descriptor(ref,int(sid),c['seed'])
        for k in c['arms']:
            e=float(abs(features[k]-desc[k][j]).max());assert e<1e-10;descriptor_error=max(e,descriptor_error)
        if (j+1)%1000==0:print(json.dumps({'recomputed_clips':j+1}),flush=True)
    lookup=np.full(len(y),-1);lookup[ids]=np.arange(len(ids));prediction_error=0.;fits=0;score_rows={k:[] for k in s['scores']};assert digest(B/'FILES.json')==c['reference_manifest_sha256'];bmanifest=json.loads((B/'FILES.json').read_text())
    for h in range(3,9):
        valid=(u>=3)&(u<=8)&~((u==6)&(y==3));v=3 if h==8 else h+1;parts={'train':np.flatnonzero(valid&(u!=h)&(u!=v)),'val':np.flatnonzero(valid&(u==v)),'diagnostic':np.flatnonzero(valid&(u==h))};d=R/f'held{h}';split=json.loads((d/'split.json').read_text())
        for k,ii in parts.items():np.testing.assert_array_equal(ii,split[k])
        tr,va,te=[parts[k] for k in ['train','val','diagnostic']];it,iv,ie=[lookup[ii] for ii in [tr,va,te]];w=np.zeros(len(tr))
        for user in np.unique(u[tr]):
            cls=np.unique(y[tr][u[tr]==user])
            for label in cls:
                mask=(u[tr]==user)&(y[tr]==label);w[mask]=1/(4*len(cls)*mask.sum())
        old=B/f'held{h}';assert digest(old/'features.npz')==bmanifest[f'held{h}/features.npz'];assert digest(old/'predictions.npz')==bmanifest[f'held{h}/predictions.npz']
        with np.load(old/'features.npz') as z:hidden=z['hidden'];np.testing.assert_array_equal(z['ids'],ids)
        with np.load(d/'models.npz') as z:models={k:z[k] for k in z.files}
        with np.load(d/'predictions.npz') as z:pred={k:z[k] for k in z.files}
        for k,expected in [('ids',te),('y',y[te]),('validation_ids',va),('validation_y',y[va])]:np.testing.assert_array_equal(pred[k],expected)
        selection=json.loads((d/'selection.json').read_text())
        for arm in c['arms']:
            x=np.concatenate([hidden,desc[arm]],1);mean=np.average(x[it],axis=0,weights=w);scale=np.sqrt(np.average((x[it]-mean)**2,axis=0,weights=w));scale=np.where(scale>1e-12,scale,1.);np.testing.assert_allclose(mean,models[arm+'_mean'],atol=1e-10);np.testing.assert_allclose(scale,models[arm+'_scale'],atol=1e-10)
            candidates=[];refs=[]
            for lam in c['lambdas']:
                model=Ridge(alpha=lam*len(tr),solver='cholesky').fit((x[it]-mean)/scale,np.eye(12)[y[tr]],sample_weight=w*len(tr));vp=model.predict((x[iv]-mean)/scale);e=float(abs(vp-pred[f'{arm}_validation_{lam}']).max());assert e<1e-7;prediction_error=max(prediction_error,e);fits+=1;candidates.append((macro(y[va],vp.argmax(1)),float(np.square(vp-np.eye(12)[y[va]]).mean()),lam));refs.append(model)
            j=min(range(3),key=lambda j:(-candidates[j][0],candidates[j][1],-candidates[j][2]));assert j==selection[arm]['selected']
            for a,b in zip(candidates,selection[arm]['candidates']):assert abs(a[0]-b['macro'])<1e-12 and abs(a[1]-b['mse'])<1e-10
            tp=refs[j].predict((x[ie]-mean)/scale);e=float(abs(tp-pred[arm]).max());assert e<1e-7;prediction_error=max(prediction_error,e);np.testing.assert_allclose(refs[j].coef_.T,models[arm+'_coef'],atol=1e-8);np.testing.assert_allclose(refs[j].intercept_,models[arm+'_bias'],atol=1e-8)
        with np.load(old/'predictions.npz') as z:
            for k in ['cnn_readout','tail_cnn']:np.testing.assert_array_equal(pred[k],z[k])
        for k in score_rows:score_rows[k].append(macro(y[te],pred[k].argmax(1)))
        print(json.dumps({'audited_held':h}),flush=True)
    for k in score_rows:np.testing.assert_allclose(score_rows[k],s['scores'][k],atol=1e-14)
    gates=[]
    for k,threshold in c['gate']['minimum_mean_gain'].items():
        delta=np.array(score_rows['coherent'])-score_rows[k];ok=bool(delta.mean()>=threshold and sum(delta>1e-12)>=4 and delta.min()>=-.01);assert ok==s['comparisons'][k]['passed'];gates.append(ok)
    assert all(gates)==s['gate_passed'];result=dict(state='passed',all_source_clips_recomputed=7100,all_tensor_locations_recomputed=7100*3*126,descriptor_values_recomputed=7100*4*162,sklearn_refits=fits,selected_models_verified=24,artifact_hashes=len(manifest),max_tensor_error=tensor_error,max_descriptor_error=descriptor_error,max_prediction_error=prediction_error,roles_and_selection_verified=True,gate_reproduced=True,seconds=time.perf_counter()-start);dump(R/'AUDIT.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
