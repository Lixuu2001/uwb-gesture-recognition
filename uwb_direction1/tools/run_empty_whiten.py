import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from audit_restored import digest,dump
from empty_whiten import template,features,full_mean
from second_order_core import fit_family,score,weights,macro
from radar_exchange import decision


def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a,dtype='<f8').tobytes()).hexdigest()


def main(a):
 c=json.loads(a.protocol.read_text());assert not a.out.exists();a.out.mkdir();dump(a.out/'PROTOCOL.json',c);identity=dict(protocol_sha256=digest(a.protocol),code_sha256={n:digest(Path(__file__).with_name(n)) for n in ['run_empty_whiten.py','empty_whiten.py','second_order_core.py','radar_exchange.py','audit_restored.py']});dump(a.out/'IDENTITY.json',identity)
 if a.cost_only:
  rng=np.random.default_rng(17);raw=rng.normal(size=(16,3,90,189));users=np.repeat([3,4,5,6],4);t=time.perf_counter();temp=template(raw,np.arange(16),users,8,c)
  for gain in [None,temp['gain'],temp['shuffled_gain']]:features(raw,gain,c)
  signal_seconds=time.perf_counter()-t;x=rng.normal(size=(4800,695));y=np.tile(np.arange(12).repeat(100),4);u=np.repeat(np.arange(3,7),1200);t=time.perf_counter();fit_family(x,y,u,c['lambdas']);fit_seconds=time.perf_counter()-t;estimate=2*(6*7100/16*signal_seconds+6*3*fit_seconds);r=dict(signal_seconds16=signal_seconds,fit_seconds=fit_seconds,estimate_seconds=estimate,passed=estimate<c['budget_seconds'],protocol_sha256=identity['protocol_sha256'],real_data_read=False);dump(a.out/'COST.json',r);print(json.dumps(r));return
 cost=json.loads(a.cost.read_text());assert cost['passed'] and cost['protocol_sha256']==identity['protocol_sha256'];old=Path(c['readout_reference']);feat=Path(c['feature_reference']);assert digest(old/'FILES.json')==c['readout_manifest_sha256'];assert digest(feat/'FILES.json')==c['feature_manifest_sha256'];rm=json.loads((old/'FILES.json').read_text());fm=json.loads((feat/'FILES.json').read_text());rawdir=Path('/workspace/uwb-restored-20260930/cache');yp=rawdir/'y_clutterremoved.npy';up=rawdir/'subj_clutterremoved.npy';assert digest(yp)=='d77b10f0fbda231246e71dfcf0b2f852863af48f80f159aed90f5f9b7c61a8bd';assert digest(up)=='5ab32ab59e5b41c05d301b76949f159ddb1e9423a749deb760585a5f19cf9480';assert digest(rawdir/'X_clutterremoved.npy')==c['input_sha256']['raw'];raw=np.load(rawdir/'X_clutterremoved.npy',mmap_mode='r');y=np.load(yp);u=np.load(up);scores={k:[] for k in c['arms']};rows=[];t=time.perf_counter()
 for h in c['users']:
  d=a.out/f'held{h}';d.mkdir();od=old/f'held{h}';fd=feat/f'held{h}'
  for n in ['models.npz','predictions.npz','selection.json','split.json']:assert digest(od/n)==rm[f'held{h}/{n}']
  assert digest(fd/'features.npz')==fm[f'held{h}/features.npz'];f=np.load(fd/'features.npz');m=np.load(od/'models.npz');sel=json.loads((od/'selection.json').read_text())['onehot'];split=json.loads((od/'split.json').read_text());tr=np.array(split['train']);va=np.array(split['val']);te=np.array(split['diagnostic']);assert not(set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te));ids=f['ids'];lookup={int(v):i for i,v in enumerate(ids)};ix={r:np.array([lookup[int(v)] for v in ii]) for r,ii in [('train',tr),('val',va),('diagnostic',te)]};empty=tr[y[tr]==11];assert set(u[empty])==set(u[tr]);temp=template(raw,empty,u,h,c);np.savez_compressed(d/'template.npz',**temp)
  models={};pred={'diagnostic_ids':te,'val_ids':va};selection={};checks=[];feature_hashes={}
  for role in ['val','diagnostic']:pred[role+'_ridge']=score(f['hidden'][ix[role]],m['mean'],m['scale'],m['bias'],m['coefs'][sel['selected']])
  for arm,gain in [('raw_blocks',None),('empty_whiten',temp['gain']),('shuffled',temp['shuffled_gain'])]:
   desc=np.concatenate([features(raw[ids[st:st+c['batch_size']]],gain,c) for st in range(0,len(ids),c['batch_size'])]);feature_hashes[arm]=array_sha(desc);x=np.concatenate([f['hidden'],desc],1);xt=x[ix['train']];mean,scale,bias,coefs=fit_family(xt,y[tr],u[tr],c['lambdas']);candidates=[]
   for lam,coef in zip(c['lambdas'],coefs):
    vp=score(x[ix['val']],mean,scale,bias,coef);candidates.append(dict(lam=lam,macro=macro(y[va],vp.argmax(1)),mse=float(np.mean((vp-np.eye(12)[y[va]])**2))));pred[arm+'_val_'+str(lam)]=vp
    ref=Ridge(alpha=lam*len(tr),solver='cholesky').fit((xt-mean)/scale,np.eye(12)[y[tr]],sample_weight=weights(y[tr],u[tr])*len(tr));rr=ref.predict((x[ix['val']]-mean)/scale);err=float(abs(rr-vp).max());assert err<1e-8;np.testing.assert_array_equal(rr.argmax(1),vp.argmax(1));checks.append(dict(arm=arm,lam=lam,max_error=err))
   j=min(range(len(candidates)),key=lambda j:(-candidates[j]['macro'],candidates[j]['mse'],-candidates[j]['lam']));selection[arm]=dict(selected=j,candidates=candidates)
   for n,v in [('mean',mean),('scale',scale),('bias',bias),('coef',coefs[j])]:models[arm+'_'+n]=v
   for role in ['val','diagnostic']:pred[role+'_'+arm]=score(x[ix[role]],mean,scale,bias,coefs[j])
  np.savez_compressed(d/'models.npz',**models);np.savez_compressed(d/'predictions.npz',**pred);np.savez_compressed(d/'descriptor_ids.npz',ids=ids);dump(d/'FEATURE_HASHES.json',feature_hashes);dump(d/'split.json',split);dump(d/'selection.json',selection);dump(d/'NUMERICAL_AUDIT.json',checks)
  with np.load(od/'predictions.npz') as truth:
   np.testing.assert_array_equal(te,truth['ids']);np.testing.assert_array_equal(va,truth['validation_ids']);np.testing.assert_allclose(pred['diagnostic_ridge'],truth['onehot']+.5,atol=1e-12,rtol=1e-12);np.testing.assert_array_equal(pred['diagnostic_ridge'].argmax(1),truth['onehot'].argmax(1));ss={k:macro(truth['y'],pred['diagnostic_'+k].argmax(1)) for k in c['arms']};np.savez_compressed(d/'evaluation.npz',diagnostic_y=truth['y'],val_y=truth['validation_y'])
  for k,v in ss.items():scores[k].append(v)
  row=dict(held=h,scores=ss,empty_training_records=len(empty),gain_range=[float(temp['gain'].min()),float(temp['gain'].max())]);rows.append(row);dump(d/'result.json',row);print(json.dumps(row),flush=True)
 comp=decision(scores,c);summary=dict(state='complete',means={k:float(np.mean(v)) for k,v in scores.items()},scores=scores,comparisons=comp,gate_passed=all(v['passed'] for v in comp.values()),rows=rows,seconds=time.perf_counter()-t,readout_fits=54,selected_readouts=18,descriptor_dimensions=567,combined_dimensions=695,encoder_training=False);dump(a.out/'SUMMARY.json',summary);dump(a.out/'FILES.json',{str(p.relative_to(a.out)):digest(p) for p in sorted(a.out.rglob('*')) if p.is_file()});print(json.dumps({k:summary[k] for k in ['means','gate_passed','seconds']}))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['protocol','out']:p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--cost-only',action='store_true');p.add_argument('--cost',type=Path);a=p.parse_args()
 with threadpool_limits(limits=1):main(a)
