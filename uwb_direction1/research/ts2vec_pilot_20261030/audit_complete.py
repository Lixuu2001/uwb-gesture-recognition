"""Independent completed-run audit:600training replay,fullfeatures,SVMrefit.

No producer import. Guards precede all arrays/models; every fit uses only the
original four source training users. No new configuration or held-label choice.
"""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import argparse,hashlib,json,random,sys,time
from pathlib import Path
import joblib
import numpy as np
B=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(4*1024*1024),b''):h.update(c)
 return h.hexdigest()
def require_complete(base):
 out=base/'run01';state=read(out/'STATE.json')
 if state.get('state')!='complete' or state.get('pid') is not None or state.get('iteration')!=600:raise RuntimeError('not_ready: incomplete/active600update pilot')
 if not state.get('all_predictions_frozen') or not state.get('diagnostic_labels_read'):raise RuntimeError('not_ready: prediction freeze/completion missing')
 assert state['users7_8_labels_used'] is False;term=read(base/'supervisor01/TERMINATION.json')
 if term['exit_code']!=0 or term['wall_timeout']:raise RuntimeError('not_ready: unsuccessful supervisor')
 assert term['supervisor_wall_seconds']<=43200 and term['child_cpu_seconds']<=43200
 frozen=read(out/'ALL_PREDICTIONS_FROZEN.json');assert set(frozen)=={'validation','diagnostic'}
 for role,h in frozen.items():assert sha(out/(role+'_predictions_unscored.npz'))==h
 done=read(out/'ENCODER_COMPLETE.json');assert done['iterations']==600 and done['initial_inclusive_average_samples']==601 and done['raw_and_average_checkpoint_sha256']==sha(out/'last.pt')
 for name in ('FILES.json','SUMMARY.json','initial.pt','first_update.pt','features.npz','svm.joblib','ITERATIONS.jsonl','SVM_TRAINING_CV_IDS.json'):
  if not (out/name).is_file():raise RuntimeError('not_ready: required artifact missing '+name)
 return term
def equal(a,b):
 import torch
 if isinstance(a,torch.Tensor):assert isinstance(b,torch.Tensor) and torch.equal(a,b)
 elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
 elif isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(tuple,list)):
  assert type(a)==type(b) and len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b
def metric(y,p):return float(sum(np.count_nonzero((y==c)&(p==c))/np.count_nonzero(y==c) for c in np.unique(y))/len(np.unique(y)))
def audit(base,output):
 term=require_complete(base);assert not output.exists();started=time.monotonic();cfg=read(base/'PROTOCOL.json');out=base/'run01';hashes={}
 def checked(p,h):assert sha(p)==h,str(p);hashes[str(p)]=h
 for n,h in read(base/'FROZEN_HASHES.json').items():checked(base/n,h)
 for n,h in read(out/'FILES.json').items():checked(out/n,h)
 identity=read(out/'IDENTITY.json');assert identity['protocol_sha256']==sha(base/'PROTOCOL.json') and identity['frozen_hashes']==read(base/'FROZEN_HASHES.json')
 import torch
 from sklearn.model_selection import StratifiedKFold
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 from training_ops import make_model,STEP,optimizer,batch_indices
 sys.path.insert(0,cfg['prototype']);from readout_adapter import fit_svm
 assert identity['torch']==torch.__version__ and identity['numpy']==np.__version__ and identity['python']==sys.version
 checked(cfg['source_manifest'],cfg['source_manifest_sha256']);checked(cfg['source_maps'],cfg['source_maps_sha256']);cache=Path(cfg['data'])/'cache'
 for n,h in cfg['input_sha256'].items():checked(cache/n,h)
 with np.load(cfg['source_manifest']) as z:ids=z['ids'];users=z['users']
 assert len(ids)==len(np.unique(ids))==6900 and set(users)==set(range(1,7))
 ix=dict(train=np.flatnonzero(np.isin(users,[3,4,5,6])),validation=np.flatnonzero(users==2),diagnostic=np.flatnonzero(users==1));assert {k:len(v) for k,v in ix.items()}==cfg['counts'];split=read(out/'split.json')
 for role,index in ix.items():np.testing.assert_array_equal(split[role],ids[index])
 subjects=np.load(cache/'subj_clutterremoved.npy',mmap_mode='r');np.testing.assert_array_equal(subjects[ids],users)
 maps=np.load(cfg['source_maps'],mmap_mode='r');raw=np.load(cache/'X_clutterremoved.npy',mmap_mode='r')
 for start in range(0,6900,64):
  x=np.array(raw[ids[start:start+64]],copy=True);rms=np.sqrt(np.mean(x*x,axis=(2,3),keepdims=True,dtype=np.float32));reference=(x/np.maximum(rms,np.float32(1e-8))).transpose(0,2,1,3).reshape(len(x),90,567);np.testing.assert_allclose(reference,maps[start:start+64],rtol=2e-5,atol=2e-6)
 train=torch.from_numpy(np.array(maps[ix['train']],copy=True));checkpoints={n:torch.load(out/(n+'.pt'),map_location='cpu',weights_only=False) for n in ('initial','first_update','last')}
 rows=[json.loads(line) for line in (out/'ITERATIONS.jsonl').read_text().splitlines()];assert [r['iteration'] for r in rows]==list(range(1,601))
 model=make_model(cfg['seed']);opt=optimizer(model);assert sum(p.numel() for p in model._net.parameters())==673472
 def check_state(saved,step):
  assert saved['identity']==identity and saved['step']==saved['n_iters']==step and saved['n_epochs']==step//293
  equal(saved['raw_model'],model._net.state_dict());equal(saved['averaged_model'],model.net.state_dict());equal(saved['optimizer'],opt.state_dict());equal(saved['torch_rng'],torch.get_rng_state());equal(saved['numpy_rng'],np.random.get_state());equal(saved['python_rng'],random.getstate());assert int(model.net.n_averaged)==step+1
 check_state(checkpoints['initial'],0)
 for step,row in enumerate(rows,1):
  index,sampler=batch_indices(step,cfg['seed']);np.testing.assert_array_equal(row['training_ids'],ids[ix['train'][index]]);assert row['sampler']==sampler
  loss=STEP(model,train[index],opt);assert float(loss.detach())==row['loss'];model.n_iters=step;model.n_epochs=step//293
  if step in (1,600):
   saved=checkpoints['first_update' if step==1 else 'last'];assert saved['sampler']==sampler;check_state(saved,step)
   if step==1:np.testing.assert_array_equal(saved['training_ids'],row['training_ids'])
 assert {int(v['step']) for v in opt.state.values()}=={600} and len(opt.state)==48
 model.net.eval();replayed={}
 with np.load(out/'features.npz') as z:
  np.testing.assert_array_equal(z['ids'],ids);np.testing.assert_array_equal(z['users'],users)
  for role,index in ix.items():
   parts=[]
   with torch.inference_mode():
    for start in range(0,len(index),16):parts.append(model.net(torch.from_numpy(np.array(maps[index[start:start+16]],copy=True))).numpy().max(axis=1))
   replayed[role]=np.concatenate(parts);np.testing.assert_array_equal(replayed[role],z[role]);assert replayed[role].shape==(len(index),320)
 labels=np.load(cache/'y_clutterremoved.npy',mmap_mode='r');yt=np.array(labels[ids[ix['train']]],copy=True)
 folds=read(out/'SVM_TRAINING_CV_IDS.json');expected=list(StratifiedKFold(5).split(replayed['train'],yt));assert len(folds)==len(expected)==5
 for row,(tr,va) in zip(folds,expected):np.testing.assert_array_equal(row['train_ids'],ids[ix['train'][tr]]);np.testing.assert_array_equal(row['validation_ids'],ids[ix['train'][va]])
 original=joblib.load(out/'svm.joblib');assert original['identity']==identity;np.testing.assert_array_equal(original['training_ids'],ids[ix['train']]);assert np.isfinite(original['cv_results']['mean_test_score']).all()
 refit=fit_svm(replayed['train'],yt);saved=original['classifier'];assert refit.get_params()==saved.get_params();np.testing.assert_array_equal(refit.support_,saved.support_);np.testing.assert_array_equal(refit.dual_coef_,saved.dual_coef_);np.testing.assert_array_equal(refit.support_vectors_,saved.support_vectors_)
 scores={}
 for role in ('validation','diagnostic'):
  predictions=refit.predict(replayed[role]);decisions=refit.decision_function(replayed[role])
  with np.load(out/(role+'_predictions_unscored.npz')) as z:np.testing.assert_array_equal(z['ids'],ids[ix[role]]);np.testing.assert_array_equal(z['predictions'],predictions);np.testing.assert_array_equal(z['decisions'],decisions)
  scores[role]=metric(np.array(labels[ids[ix[role]]],copy=True),predictions)
 summary=read(out/'SUMMARY.json');reference=cfg['reference_held1_macro'];assert scores==summary['scores'] and summary['reference_held1_macro']==reference
 eligible=bool(scores['diagnostic']-reference+1e-12>=.005);assert summary['known_baseline_expansion_eligible']==eligible and summary['predeclared_route']==('known_baseline_source_confirmation' if eligible else 'close_exact_configuration')
 assert summary['users7_8_labels_used'] is False and summary['independent_validation'] is False and summary['novel_contribution_validated'] is False
 for p,h in hashes.items():assert sha(p)==h,p
 result=dict(state='pass',scores=scores,reference_held1_macro=reference,known_baseline_expansion_eligible=eligible,source_preprocessing_records_verified=6900,training_updates_replayed_exact=600,training_sample_appearances_verified=9600,initial_first_last_raw_average_optimizer_all_RNG_exact=True,all6900representations_bitwise_equal=True,original_training_only_fivefold_SVM_refit_support_dual_predictions_exact=True,validation_and_diagnostic_predictions_verified=2200,artifact_hashes=hashes,training_termination=term,auditor_sha256=sha(__file__),audit_wall_seconds=time.monotonic()-started,users7_8_labels_used=False,independent_validation=False,novel_contribution_validated=False,limitations='Read-only completed-run audit with fixed training-only reconstruction/refit; no new candidate/Cgrid/held-label selection. Source users historically developed, not independent confirmation.')
 output.write_text(json.dumps(result,indent=2));print(json.dumps(dict(state='pass',output=str(output))))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();audit(B,args.out)
