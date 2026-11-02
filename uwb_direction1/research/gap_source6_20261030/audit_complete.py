"""Read-only completed GAP confirmation audit; rejects partial runs before arrays.

No training-driver imports, model fitting or new model/configuration selection.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import argparse,hashlib,json,math,random,sys,time
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent

def read(p): return json.loads(Path(p).read_text())
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
 return h.hexdigest()
def require_complete(base):
 run=base/'run01';state=read(run/'STATE.json')
 if state.get('state')!='complete' or state.get('pid') is not None: raise RuntimeError('not_ready: GAP training not complete')
 if state.get('all_new_diagnostic_predictions_frozen') is not True: raise RuntimeError('not_ready: prediction freeze absent')
 if set(state.get('completed_folds',[]))!=set(range(1,7)): raise RuntimeError('not_ready: missing fold')
 term=read(base/'supervisor01/TERMINATION.json')
 if term['exit_code']!=0 or term['wall_timeout']: raise RuntimeError('not_ready: unsuccessful training supervisor')
 assert term['supervisor_wall_seconds']<=43200 and term['child_cpu_seconds']<=43200
 frozen=read(run/'ALL_NEW_PREDICTIONS_FROZEN.json');assert set(frozen)==set(map(str,range(2,7)))
 for held in range(2,7):
  folder=run/f'held{held}';done=read(folder/'FOLD_COMPLETE.json')
  assert done['held']==held and done['epochs']==64
  assert digest(folder/'predictions_unscored.npz')==frozen[str(held)]==done['prediction_sha256']
  assert digest(folder/'best.pt')==done['checkpoint_sha256']
 return term
def choose(rows): return sorted(rows,key=lambda r:(-r['validation_macro'],r['validation_loss'],r['epoch']))[0]
def normalized_history(rows,reused):
 return [dict(r,validation_loss=r['validation_bce']) for r in rows] if reused else rows
def metric(y,p):
 assert len(y)==len(p) and len(y)>0
 return float(sum(np.count_nonzero((y==c)&(p==c))/np.count_nonzero(y==c) for c in np.unique(y))/len(np.unique(y)))
def gate(gains): return bool(np.mean(gains)>=.01 and np.count_nonzero(np.asarray(gains)>0)>=4 and min(gains)>=-.02)
def split_indices(split,ids,users,held):
 val=held%6+1;ix=dict(train=np.where((users!=held)&(users!=val))[0],validation=np.where(users==val)[0],diagnostic=np.where(users==held)[0])
 for role,index in ix.items(): np.testing.assert_array_equal(split[role],ids[index])
 assert not (set(split['train'])&set(split['validation']) or set(split['train'])&set(split['diagnostic']) or set(split['validation'])&set(split['diagnostic']))
 return ix
def preprocess(x):
 rms=np.sqrt(np.mean(np.square(x),axis=(2,3),keepdims=True,dtype=np.float32))
 return (x/np.maximum(rms,np.float32(1e-8))).transpose(0,2,1,3).reshape(len(x),90,567)
def nested_equal(a,b):
 import torch
 if isinstance(a,torch.Tensor): assert isinstance(b,torch.Tensor) and torch.equal(a,b)
 elif isinstance(a,dict):
  assert set(a)==set(b)
  for k in a: nested_equal(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b): nested_equal(x,y)
 else: assert a==b
def optimizer_check(model,saved,steps,constructor):
 import torch
 params=list(model.parameters());assert saved['count']==steps and saved['alpha']==.5 and saved['k']==6
 fast=saved['fast'];assert len(fast['param_groups'])==1
 group=fast['param_groups'][0];assert group['params']==list(range(len(params))) and group['lr']==.001 and group['weight_decay']==.0001
 assert set(fast['state'])==set(range(len(params))) and {int(v['step']) for v in fast['state'].values()}=={steps}
 assert set(saved['slow'])==(set(range(len(params))) if steps>=6 else set())
 for i,p in enumerate(params):
  for k in ('exp_avg','exp_avg_sq'): assert fast['state'][i][k].shape==p.shape and torch.isfinite(fast['state'][i][k]).all()
  if steps>=6: assert saved['slow'][i].shape==p.shape and torch.isfinite(saved['slow'][i]).all()
 restored=constructor(model.parameters());restored.load_state_dict(saved);nested_equal(saved,restored.state_dict())
def audit(base,output):
 term=require_complete(base)  # Precedes torch import and every array/checkpoint load.
 assert not output.exists()
 import torch
 from torch.nn import functional as F
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 started=time.monotonic();cfg=read(base/'PROTOCOL.json');run=base/'run01';identity=read(run/'IDENTITY.json');frozen=read(base/'FROZEN_HASHES.json');hashes={}
 def checked(path,expected):
  assert digest(path)==expected,str(path)
  hashes[str(path)]=expected
 for n,h in frozen.items(): checked(base/n,h)
 for n,h in read(base/'AUDIT_INPUT_HASHES.json').items(): checked(Path(n),h)
 for n,h in read(base/'AUDIT_HASHES.json').items(): checked(base/n,h)
 assert identity['frozen_hashes']==frozen and identity['protocol_sha256']==digest(base/'PROTOCOL.json')
 assert identity['torch']==torch.__version__ and identity['numpy']==np.__version__ and identity['python']==sys.version
 for n,h in read(run/'FILES.json').items(): checked(run/n,h)
 checked(cfg['source_manifest'],cfg['source_manifest_sha256']);checked(cfg['source_maps'],cfg['source_maps_sha256'])
 cache=Path(cfg['data'])/'cache'
 for n,h in cfg['input_sha256'].items(): checked(cache/n,h)
 assert cfg['gate']==dict(mean_gap_minus_localcnn_minimum=.01,positive_users_minimum=4,worst_user_difference_minimum=-.02)
 assert cfg['epochs']==64 and cfg['new_held_users']==[2,3,4,5,6] and cfg['threads']==1
 assert read(cfg['upstream_audit'])['state']==read(cfg['localcnn_audit'])['state']=='pass'
 with np.load(cfg['source_manifest']) as z: ids=z['ids'];users=z['users']
 assert len(ids)==len(np.unique(ids))==6900 and set(users)==set(range(1,7))
 source=np.load(cfg['source_maps'],mmap_mode='r');assert source.shape==(6900,90,567) and source.dtype==np.float32
 raw=np.load(cache/'X_clutterremoved.npy',mmap_mode='r')
 for offset in range(0,len(ids),64): np.testing.assert_allclose(source[offset:offset+64],preprocess(np.array(raw[ids[offset:offset+64]],copy=True)),rtol=2e-5,atol=2e-6)
 # Only source1..6 indices are used after the completion/freeze guard.
 labels=np.load(cache/'y_clutterremoved.npy',mmap_mode='r');summary=read(run/'SUMMARY.json')
 assert summary['state']=='complete_source_confirmation' and summary['users7_8_labels_used'] is False
 sys.path.insert(0,str(Path(cfg['reused_held1']['directory']).parents[1]))
 from adapter import make_model,StableLookahead
 rows=[];details=[];prior_cnn=read(cfg['localcnn_audit'])
 for held in range(1,7):
  folder=Path(cfg['reused_held1']['directory']) if held==1 else run/f'held{held}'
  split=read(folder.parent/'split.json') if held==1 else read(folder/'split.json')
  ix=split_indices(split,ids,users,held);assert [len(ix[k]) for k in ('train','validation','diagnostic')]==cfg['counts'][str(held)]
  checkpoints={n:torch.load(folder/f'{n}.pt',map_location='cpu',weights_only=False) for n in ('initial','first_update','best','last')}
  initial,first,best,last=[checkpoints[k] for k in ('initial','first_update','best','last')]
  expected_identity=read(folder.parent/'IDENTITY.json') if held==1 else identity
  assert all(c['identity']==expected_identity for c in checkpoints.values())
  if held!=1: assert all(c['held']==held for c in checkpoints.values())
  history=normalized_history([json.loads(line) for line in (folder/'epochs.jsonl').read_text().splitlines()],held==1)
  assert [r['epoch'] for r in history]==list(range(1,65));chosen=choose(history)
  assert initial['epoch']==0 and last['epoch']==64 and best['epoch']==chosen['epoch']
  assert best['validation_macro']==chosen['validation_macro'] and best['validation_loss']==chosen['validation_loss']
  assert tuple(last['best'])==(chosen['validation_macro'],-chosen['validation_loss'],-chosen['epoch'])
  done=read(folder/('ARM_COMPLETE.json' if held==1 else 'FOLD_COMPLETE.json'));assert done['epochs']==64 and done['selected_epoch']==best['epoch']
  checked(folder/'best.pt',done['checkpoint_sha256']);checked(folder/'predictions_unscored.npz',done['prediction_sha256'])
  model=make_model('gap',cfg['seed']);assert sum(p.numel() for p in model.parameters())==617612
  assert all(torch.equal(v,initial['model'][k]) for k,v in model.state_dict().items())
  assert any(not torch.equal(v,first['model'][k]) for k,v in initial['model'].items())
  expected=torch.randperm(len(ix['train']),generator=torch.Generator().manual_seed(cfg['seed']+1)).numpy()[:64]
  np.testing.assert_array_equal(first['training_ids'],ids[ix['train'][expected]])
  steps=64*math.ceil(len(ix['train'])/64);optimizer_check(model,first['optimizer'],1,StableLookahead);optimizer_check(model,last['optimizer'],steps,StableLookahead)
  for c in checkpoints.values(): assert all(torch.isfinite(v).all() for v in c['model'].values() if v.is_floating_point())
  assert all(int(v)==steps for k,v in last['model'].items() if k.endswith('num_batches_tracked'))
  torch.set_rng_state(last['torch_rng']);np.random.RandomState().set_state(last['numpy_rng']);random.Random().setstate(last['python_rng'])
  if best['epoch']==64: nested_equal(best['model'],last['model'])
  model.load_state_dict(best['model']);model.eval()
  def replay(index):
   parts=[]
   with torch.inference_mode():
    for offset in range(0,len(index),64): parts.append(model(torch.from_numpy(np.array(source[index[offset:offset+64]],copy=True))).numpy())
   value=np.concatenate(parts);assert np.isfinite(value).all();return value
  validation=replay(ix['validation']);yv=np.array(labels[ids[ix['validation']]],copy=True)
  assert abs(metric(yv,validation.argmax(1))-best['validation_macro'])<1e-12
  bce=float(F.binary_cross_entropy_with_logits(torch.from_numpy(validation),F.one_hot(torch.from_numpy(yv).long(),12).float()))
  np.testing.assert_allclose(bce,best['validation_loss'],rtol=1e-4,atol=1e-5)
  diagnostic=replay(ix['diagnostic']);yd=np.array(labels[ids[ix['diagnostic']]],copy=True)
  with np.load(folder/'predictions_unscored.npz') as z:
   np.testing.assert_array_equal(z['ids'],ids[ix['diagnostic']]);np.testing.assert_allclose(diagnostic,z['logits'],rtol=1e-4,atol=1e-5);np.testing.assert_array_equal(diagnostic.argmax(1),z['logits'].argmax(1));score=metric(yd,z['logits'].argmax(1))
  ref=cfg['cnn_references'][str(held)];checked(ref['path'],ref['sha256'])
  assert prior_cnn['artifact_hashes'][ref['path']]==ref['sha256']
  with np.load(ref['path']) as z: np.testing.assert_array_equal(z['ids'],ids[ix['diagnostic']]);baseline=metric(yd,z['logits'].argmax(1))
  old=next(r for r in prior_cnn['rows'] if r['held']==held);assert abs(baseline-old['local'])<1e-12
  row=dict(held=held,gap=score,localcnn=baseline,difference=score-baseline);original=next(r for r in summary['rows'] if r['held']==held)
  for k in ('gap','localcnn','difference'): assert abs(row[k]-original[k])<1e-12
  rows.append(row);details.append(dict(held=held,selected_epoch=best['epoch'],optimizer_steps=steps,validation_forward_records=len(yv),diagnostic_logits_replayed=len(yd),initial_first_optimizer_rng_bn_verified=True))
  del model,checkpoints
 gains=[r['difference'] for r in rows]
 for k in ('gap','localcnn'): assert abs(np.mean([r[k] for r in rows])-summary['means'][k])<1e-12
 assert abs(np.mean(gains)-summary['mean_difference'])<1e-12 and summary['wins']==sum(g>0 for g in gains)
 assert summary['descriptive_continuation_gate']==gate(gains)
 assert summary['new_epochs']==320 and summary['new_trajectories']==5 and summary['reused_fold']==1
 assert summary['cpu_seconds']<=43200 and summary['wall_seconds']<=43200
 for p,h in hashes.items(): assert digest(p)==h,'Changed during read-only audit: '+p
 result=dict(state='pass',scope='completed_gap_source_sixfold_read_only',rows=rows,folds=details,descriptive_continuation_gate=gate(gains),artifact_hashes=hashes,source_preprocessing_records_verified=6900,training_termination=term,auditor_sha256=digest(Path(__file__)),audit_wall_seconds=time.monotonic()-started,users7_8_labels_used=False,independent_validation=False,novel_contribution_validated=False,limitation='All GAP selected-validation forwards and diagnostic logits replayed. CNN reference hashes/metrics checked against prior independent full-forward audit, CNN forwards not repeated. Historical nonselected GAP epochs not retrained/replayed.')
 assert not output.exists();output.write_text(json.dumps(result,indent=2));print(json.dumps(dict(state='pass',report=str(output))))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();audit(BASE,args.out)
