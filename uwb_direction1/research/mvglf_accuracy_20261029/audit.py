"""Independent partial/complete audit; partial never indexes diagnostic labels.
Read-only snapshots of atomic checkpoints avoid altering the live training run.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,hashlib,io,json,math,resource,time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import recall_score
from model import MVGLF
B=Path(__file__).resolve().parent;R=B/'run01'
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def snapshot(p):
 data=p.read_bytes();return torch.load(io.BytesIO(data),map_location='cpu',weights_only=False),hashlib.sha256(data).hexdigest()
def atomic(p,v):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2));q.replace(p)
def selection(rows):return max(rows,key=lambda r:(r['validation_macro'],-r['validation_loss'],-r['epoch']))
def audit(partial,output):
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 resource.setrlimit(resource.RLIMIT_CPU,(870,890));started=time.monotonic();checks=0
 cfg=json.loads((B/'PROTOCOL.json').read_text());ident=json.loads((R/'IDENTITY.json').read_text());state=json.loads((R/'STATE.json').read_text())
 assert ident['protocol_sha256']==digest(B/'PROTOCOL.json');assert ident['model_sha256']==digest(B/'model.py');assert ident['runner_sha256']==digest(B/'train.py');checks+=3
 for name,h in json.loads((B/'SOURCE_HASHES.json').read_text()).items():assert digest(B/name)==h;checks+=1
 assert digest(cfg['source_manifest'])==cfg['source_manifest_sha256'];checks+=1
 with np.load(cfg['source_manifest']) as z:ids=z['ids'];users=z['users']
 assert set(users)==set(range(1,7));assert len(ids)==6900;checks+=2
 splits=json.loads((R/'split.json').read_text());lookup={int(v):i for i,v in enumerate(ids)}
 ix={key:np.array([lookup[int(v)] for v in splits[key]]) for key in ['train','validation','diagnostic']}
 for name,user_set,n in [('train',{3,4,5,6},4700),('validation',{2},1000),('diagnostic',{1},1200)]:
  assert len(ix[name])==n and set(users[ix[name]])==user_set;checks+=1
 assert not(set(splits['train'])&set(splits['validation']) or set(splits['train'])&set(splits['diagnostic']) or set(splits['validation'])&set(splits['diagnostic']));checks+=1
 # Byte-only integrity checks do not semantically inspect unused gesture labels.
 for n,h in cfg['input_sha256'].items():assert digest(Path(cfg['data'])/'cache'/n)==h;checks+=1
 cacheinfo=json.loads((R/'SOURCE_CACHE.json').read_text());assert digest(R/'source_maps.npy')==cacheinfo['sha256'];checks+=1
 reports={};available=[]
 for arm in cfg['arms']:
  d=R/arm
  if not (d/'last.pt').exists():
   if partial:reports[arm]={'state':'no_completed_epoch_yet'};continue
   raise AssertionError('missing epoch checkpoint')
  last,lh=snapshot(d/'last.pt');best,bh=snapshot(d/'best.pt');records=[json.loads(v) for v in (d/'epochs.jsonl').read_text().splitlines() if v.strip()]
  ep=last['epoch'];records=[r for r in records if r['epoch']<=ep]
  if len(records)<ep or best['epoch']>ep:
   if partial:reports[arm]={'state':'atomic_snapshot_crossed_epoch_boundary_retry_later'};continue
   raise AssertionError('inconsistent completed epoch records')
  assert [r['epoch'] for r in records]==list(range(1,ep+1));checks+=1
  chosen=selection(records);assert best['epoch']==chosen['epoch'];assert best['validation_macro']==chosen['validation_macro'];assert best['validation_loss']==chosen['validation_loss'];checks+=3
  assert tuple(last['best'])==(chosen['validation_macro'],-chosen['validation_loss'],-chosen['epoch']);checks+=1
  assert last['identity']==ident==best['identity'];checks+=1
  steps={int(v['step']) for v in last['optimizer']['state'].values()};assert steps=={ep*math.ceil(4700/32)};checks+=1
  assert all(torch.isfinite(v).all() for v in last['model'].values() if v.is_floating_point());assert torch.is_tensor(last['torch_rng']) and last['torch_rng'].numel()>0;assert len(last['numpy_rng'])==5;checks+=3
  assert all(int(v)==ep*147 for k,v in last['model'].items() if k.endswith('num_batches_tracked'));checks+=1
  if best['epoch']==ep:assert all(torch.equal(v,best['model'][k]) for k,v in last['model'].items());checks+=1
  available.append(arm);reports[arm]=dict(state='verified',last_epoch=ep,selected_epoch=best['epoch'],validation_macro=best['validation_macro'],last_checkpoint_sha256=lh,best_checkpoint_sha256=bh,optimizer_steps=int(next(iter(steps))),optimizer_rng_bn_state_verified=True)
  if not partial:assert ep==cfg['epochs'];checks+=1
 result=dict(state='pass',scope='partial_no_diagnostic_label_access' if partial else 'complete',observed_unix=time.time(),checks_passed=checks,arms=reports,runner_state_observed=state['state'],training_process_pid_observed=state.get('pid'),users7_8_labels_used=False,diagnostic_labels_indexed_by_auditor=False,scientific_accuracy_claim=False,identity=ident,auditor_sha256=digest(Path(__file__)))
 if not partial:
  assert state['state']=='complete';assert set(available)==set(cfg['arms']);checks+=2
  summary=json.loads((R/'SUMMARY.json').read_text())
  for p,h in json.loads((R/'FILES.json').read_text()).items():assert digest(R/p)==h;checks+=1
  # Require all predictions to exist, verify their preregistered freezes first.
  for arm in cfg['arms']:
   complete=json.loads((R/arm/'ARM_COMPLETE.json').read_text());assert digest(R/arm/'predictions_unscored.npz')==complete['prediction_sha256'];assert digest(R/arm/'best.pt')==complete['checkpoint_sha256'];checks+=2
  # Only now index diagnostic labels, after both complete/frozen prediction sets.
  labels=np.load(Path(cfg['data'])/'cache/y_clutterremoved.npy',mmap_mode='r');yt=np.asarray(labels[ids[ix['diagnostic']]]).copy()
  source=np.load(R/'source_maps.npy',mmap_mode='r');scores={};yv=np.asarray(labels[ids[ix['validation']]]).copy()
  for arm in cfg['arms']:
   z,_=snapshot(R/arm/'best.pt');m=MVGLF(arm).to(memory_format=torch.channels_last).eval();m.load_state_dict(z['model']);preds=[]
   val_predictions=[]
   with torch.no_grad():
    for b in range(0,len(yv),32):
     raw=np.array(source[ix['validation'][b:b+32]],copy=True);val_predictions.append(m(torch.from_numpy(raw).contiguous(memory_format=torch.channels_last)))
   vp=torch.cat(val_predictions);vm=float(recall_score(yv,vp.argmax(1).numpy(),labels=np.unique(yv),average='macro',zero_division=0));vl=float(torch.nn.functional.cross_entropy(vp,torch.from_numpy(yv).long()))
   assert abs(vm-z['validation_macro'])<1e-12;np.testing.assert_allclose(vl,z['validation_loss'],rtol=1e-4,atol=1e-5);checks+=2
   with torch.no_grad():
    for b in range(0,len(yt),32):
     raw=np.array(source[ix['diagnostic'][b:b+32]],copy=True);preds.append(m(torch.from_numpy(raw).contiguous(memory_format=torch.channels_last)).numpy())
   replay=np.concatenate(preds)
   with np.load(R/arm/'predictions_unscored.npz') as p:
    np.testing.assert_array_equal(p['ids'],ids[ix['diagnostic']]);np.testing.assert_allclose(replay,p['logits'],rtol=1e-4,atol=1e-5);np.testing.assert_array_equal(replay.argmax(1),p['logits'].argmax(1));checks+=3
    scores[arm]=float(recall_score(yt,p['logits'].argmax(1),labels=np.unique(yt),average='macro',zero_division=0));assert abs(scores[arm]-summary['scores'][arm])<1e-12;checks+=1
   del m
  with np.load(cfg['minirocket_reference']) as z:
   ii=z['users']==1;np.testing.assert_array_equal(z['ids'][ii],ids[ix['diagnostic']]);np.testing.assert_array_equal(z['y'][ii],yt);scores['minirocket']=float(recall_score(yt,z['minirocket'][ii],labels=np.unique(yt),average='macro',zero_division=0));assert abs(scores['minirocket']-summary['scores']['minirocket'])<1e-12;checks+=3
  expected=bool(scores['full']-scores['local']>=.005 and scores['full']>=scores['minirocket']);assert summary['full_gate_passed']==expected;checks+=1
  assert summary['cumulative_wall_seconds']<=43200 and summary['cumulative_cpu_seconds']<=129600;checks+=1
  result.update(checks_passed=checks,diagnostic_labels_indexed_by_auditor=True,scores=scores,full_gate_passed=expected,independent_validation=False,novel_contribution_validated=False)
 result['audit_wall_seconds']=time.monotonic()-started;atomic(output,result);print(json.dumps(result))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert not a.out.exists(),'use new audit output path';audit(a.partial,a.out)
