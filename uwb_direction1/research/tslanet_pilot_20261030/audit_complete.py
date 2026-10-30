"""Independent read-only TSLANet audit; no training-driver imports or fitting.

Completion/freeze/supervisor guards precede array and checkpoint loads. SSL
validation is stochastic: its recorded losses/selection and model transfer are
checked, but its original masked validation forwards are NOT replayed. Both
classifier selected-validation and diagnostic forwards are replayed in full.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import argparse, hashlib, json, math, random, sys, time
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
PHASES=(('pretrained','pretrain',50),('pretrained','classification',100),('scratch','classification',100))

def read(path): return json.loads(Path(path).read_text())
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as stream:
  for block in iter(lambda:stream.read(4*1024*1024),b''): h.update(block)
 return h.hexdigest()
def require_phase(base,arm,kind,epochs):
 if (arm,kind,epochs) not in PHASES: raise RuntimeError('not_ready: unregistered phase')
 folder=base/'run01'/arm/kind
 if not (folder/'PHASE_COMPLETE.json').is_file(): raise RuntimeError('not_ready: phase incomplete')
 done=read(folder/'PHASE_COMPLETE.json')
 assert (done['arm'],done['phase'],done['epochs'])==(arm,kind,epochs)
 assert digest(folder/'best.pt')==done['checkpoint_sha256']
 for n in ('initial.pt','first_update.pt','last.pt','epochs.jsonl'): assert (folder/n).is_file(),n
 rows=[json.loads(line) for line in (folder/'epochs.jsonl').read_text().splitlines()]
 assert [r['epoch'] for r in rows]==list(range(1,epochs+1))
 assert all(r['arm']==arm and r['phase']==kind and r['training_records_used']==4688 for r in rows)
 assert all(math.isfinite(r['training_loss']) and math.isfinite(r['validation_loss']) for r in rows)
 selected=choose(rows,kind)
 assert selected['epoch']==done['selected_epoch']
 return folder,rows,selected
def require_complete(base):
 run=base/'run01';state=read(run/'STATE.json')
 if state.get('state')!='complete' or state.get('pid') is not None: raise RuntimeError('not_ready: training incomplete')
 if state.get('completed_arms')!=['pretrained','scratch'] or state.get('diagnostic_labels_read') is not True: raise RuntimeError('not_ready: arms incomplete')
 assert state['users7_8_labels_used'] is False
 frozen=read(run/'ALL_PREDICTIONS_FROZEN.json');assert set(frozen)=={'pretrained','scratch'}
 for arm in ('pretrained','scratch'):
  done=read(run/arm/'ARM_COMPLETE.json')
  assert done['arm']==arm and done['classification_epochs']==100 and done['pretraining_epochs']==(50 if arm=='pretrained' else 0)
  assert digest(run/arm/'predictions_unscored.npz')==frozen[arm]==done['prediction_sha256']
  assert digest(run/arm/'classification/best.pt')==done['checkpoint_sha256']
 for args in PHASES: require_phase(base,*args)
 term=read(base/'supervisor01/TERMINATION.json')
 if term['exit_code']!=0 or term['wall_timeout']: raise RuntimeError('not_ready: unsuccessful supervisor')
 assert term['supervisor_wall_seconds']<=43200 and term['child_cpu_seconds']<=43200
 assert (run/'FILES.json').is_file() and (run/'SUMMARY.json').is_file()
 return term
def choose(rows,kind):
 return min(rows,key=lambda r:(r['validation_loss'],r['epoch']) if kind=='pretrain' else (-r['validation_macro'],r['validation_loss'],r['epoch']))
def metric(y,p):
 assert len(y)==len(p)>0
 return float(sum(np.count_nonzero((y==c)&(p==c))/np.count_nonzero(y==c) for c in np.unique(y))/len(np.unique(y)))
def decision(scores,reference):
 p,s=scores['pretrained'],scores['scratch'];tol=1e-12
 m=bool(p-s+tol>=.005 and p+tol>=reference-.005);b=bool(s+tol>=reference+.005);a=bool(p+tol>=reference+.005)
 route='pretrained_mechanism' if m else 'scratch_baseline' if b else 'pretrained_baseline' if a else 'close_exact_configuration'
 return dict(mechanism_expansion_eligible=m,scratch_baseline_expansion_eligible=b,pretrained_baseline_expansion_eligible=a,predeclared_route=route)
def role_indices(split,ids,users):
 assert len(ids)==len(np.unique(ids))==6900 and set(users)==set(range(1,7))
 ix=dict(train=np.flatnonzero(np.isin(users,[3,4,5,6])),validation=np.flatnonzero(users==2),diagnostic=np.flatnonzero(users==1))
 assert {k:len(v) for k,v in ix.items()}==dict(train=4700,validation=1000,diagnostic=1200)
 assert set(split)==set(ix)
 for role,index in ix.items(): np.testing.assert_array_equal(split[role],ids[index])
 return ix
def nested_equal(a,b):
 import torch
 if isinstance(a,torch.Tensor): assert isinstance(b,torch.Tensor) and torch.equal(a,b)
 elif isinstance(a,np.ndarray): np.testing.assert_array_equal(a,b)
 elif isinstance(a,dict):
  assert set(a)==set(b)
  for k in a: nested_equal(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert type(a)==type(b) and len(a)==len(b)
  for x,y in zip(a,b): nested_equal(x,y)
 else: assert a==b
def checked(path,expected,hashes):
 path=Path(path);assert digest(path)==expected,str(path);hashes[str(path)]=expected
def identities(base):
 cfg=read(base/'PROTOCOL.json');identity=read(base/'run01/IDENTITY.json');frozen=read(base/'FROZEN_HASHES.json');hashes={}
 for n,h in frozen.items(): checked(base/n,h,hashes)
 assert identity['protocol_sha256']==digest(base/'PROTOCOL.json') and identity['frozen_hashes']==frozen
 import torch
 assert identity['torch']==torch.__version__ and identity['numpy']==np.__version__ and identity['python']==sys.version
 checked(cfg['source_manifest'],cfg['source_manifest_sha256'],hashes)
 return cfg,identity,hashes
def optimizer_check(model,saved,steps,unused):
 import torch
 named=list(model.named_parameters());active={i for i,(n,p) in enumerate(named) if n not in unused}
 assert len(saved['param_groups'])==1
 group=saved['param_groups'][0]
 assert group['params']==list(range(len(named))) and group['lr']==.001 and group['weight_decay']==.0001
 assert tuple(group['betas'])==(.9,.999) and group['eps']==1e-8 and group['amsgrad'] is False and group['maximize'] is False
 assert set(saved['state'])==active and {int(v['step']) for v in saved['state'].values()}=={steps}
 for i in active:
  for key in ('exp_avg','exp_avg_sq'):
   value=saved['state'][i][key];assert value.shape==named[i][1].shape and value.dtype==named[i][1].dtype and torch.isfinite(value).all()
  assert torch.all(saved['state'][i]['exp_avg_sq']>=0)
 restored=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);restored.load_state_dict(saved);nested_equal(saved,restored.state_dict())
 return len(active)
def verify_stage(base,args,cfg,identity,ids,ix,hashes):
 import torch
 sys.path.insert(0,cfg['prototype']);from author_core import make_model
 arm,kind,epochs=args;folder,rows,chosen=require_phase(base,*args)
 for path in sorted(folder.iterdir()):
  assert path.is_file() and path.suffix!='.tmp';hashes[str(path)]=digest(path)
 cps={n:torch.load(folder/(n+'.pt'),map_location='cpu',weights_only=False) for n in ('initial','first_update','best','last')}
 initial,first,best,last=(cps[n] for n in ('initial','first_update','best','last'))
 assert all(c['identity']==identity and c['arm']==arm and c['phase']==kind for c in cps.values())
 assert initial['epoch']==0 and last['epoch']==epochs and best['epoch']==chosen['epoch']
 assert best['validation_loss']==chosen['validation_loss'] and best['validation_macro']==chosen['validation_macro']
 if kind=='pretrain':
  assert all(r['validation_macro'] is None for r in rows);key=(chosen['validation_loss'],chosen['epoch'])
 else:
  assert all(0<=r['validation_macro']<=1 for r in rows);key=(-chosen['validation_macro'],chosen['validation_loss'],chosen['epoch'])
 assert tuple(last['best_selection'])==key
 model,_=make_model(cfg['seed']);assert sum(p.numel() for p in model.parameters())==4132879
 if arm=='pretrained' and kind=='classification':
  ssl=base/'run01/pretrained/pretrain/best.pt';origin=dict(type='selected_pretrain_model_only_fresh_optimizer_and_seed',path=str(ssl),sha256=digest(ssl))
  transferred=torch.load(ssl,map_location='cpu',weights_only=False);nested_equal(initial['model'],transferred['model']);del transferred
 else:
  origin=dict(type='seeded_author_UEA_initialization',seed=cfg['seed']);nested_equal(initial['model'],model.state_dict())
 assert initial['origin']==best['origin']==origin
 expected_shapes=model.state_dict();unused=set(cfg['unused_parameters'][kind])
 assert unused==({'input_layer.weight','input_layer.bias','head.weight','head.bias'} if kind=='pretrain' else {'input_layer.weight','input_layer.bias'})
 for c in cps.values():
  assert set(c['model'])==set(expected_shapes)
  for n,v in c['model'].items():
   assert v.shape==expected_shapes[n].shape and v.dtype==expected_shapes[n].dtype and torch.isfinite(v).all()
  for n in unused: assert torch.equal(c['model'][n],initial['model'][n])
 changed=sum(not torch.equal(v,first['model'][n]) for n,v in initial['model'].items());assert changed==len(expected_shapes)-len(unused)
 order=torch.randperm(4700,generator=torch.Generator().manual_seed(cfg['seed']+1)).numpy()[:16]
 np.testing.assert_array_equal(first['training_ids'],ids[ix['train'][order]])
 steps=epochs*(4700//16);active=optimizer_check(model,first['optimizer'],1,unused);optimizer_check(model,last['optimizer'],steps,unused)
 # RNGs are well-formed and reloadable. No claim of full trajectory replay.
 rng=last['torch_rng'];assert rng.dtype==torch.uint8 and rng.shape==torch.get_rng_state().shape;torch.set_rng_state(rng);assert torch.equal(torch.get_rng_state(),rng)
 np.random.RandomState().set_state(last['numpy_rng']);random.Random().setstate(last['python_rng'])
 if best['epoch']==epochs: nested_equal(best['model'],last['model'])
 report=dict(arm=arm,phase=kind,epochs=epochs,selected_epoch=best['epoch'],optimizer_steps=steps,active_optimizer_parameter_tensors=active,first_changed_model_tensors=changed,first_training_ids_verified=16,initial_origin_verified=True,selected_history_verified=True,optimizer_reset_and_reload_verified=True,rng_reload_verified=True,unused_parameters_unchanged=True,phase_payload_files=len(list(folder.iterdir())),ssl_masked_validation_forward_replayed=False)
 return report,best
def configure_torch():
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
def audit_ssl_phase(base,output):
 require_phase(base,*PHASES[0]) # Before array/checkpoint loads; can run alongside next phase.
 assert not output.exists();started=time.monotonic();configure_torch();cfg,identity,hashes=identities(base)
 with np.load(cfg['source_manifest']) as z: ids=z['ids'];users=z['users']
 ix=role_indices(read(base/'run01/split.json'),ids,users)
 details,_=verify_stage(base,PHASES[0],cfg,identity,ids,ix,hashes)
 for p,h in hashes.items(): assert digest(p)==h,'changed during immutable phase audit: '+p
 result=dict(state='pass',scope='immutable_SSL_phase_artifact_history_audit_only',phase=details,artifact_hashes=hashes,auditor_sha256=digest(__file__),audit_wall_seconds=time.monotonic()-started,real_map_or_label_values_read=False,real_forward_or_fit=False,users7_8_labels_used=False,limitations='Recorded SSL loss history/selection, finite states, AdamW counts/reload, seeded initialization and unused parameters checked. Original stochastic SSL validation losses NOT independently reconstructed. This is not classifier accuracy evidence, complete pilot audit, or full-job resume snapshot.')
 output.write_text(json.dumps(result,indent=2));print(json.dumps(dict(state='pass',scope=result['scope'],output=str(output))))
def preprocess(raw):
 rms=np.sqrt(np.mean(np.square(raw),axis=(2,3),keepdims=True,dtype=np.float32))
 return (raw/np.maximum(rms,np.float32(1e-8))).transpose(0,2,1,3).reshape(len(raw),90,567)
def audit(base,output):
 term=require_complete(base) # Before array/checkpoint loads and torch import.
 assert not output.exists();started=time.monotonic();configure_torch();cfg,identity,hashes=identities(base);run=base/'run01'
 import torch
 from torch.nn import functional as F
 for n,h in read(base/'AUDIT_HASHES.json').items(): checked(base/n,h,hashes)
 for n,h in read(run/'FILES.json').items(): checked(run/n,h,hashes)
 checked(cfg['source_maps'],cfg['source_maps_sha256'],hashes);cache=Path(cfg['data'])/'cache'
 for n,h in cfg['input_sha256'].items(): checked(cache/n,h,hashes)
 with np.load(cfg['source_manifest']) as z: ids=z['ids'];users=z['users']
 ix=role_indices(read(run/'split.json'),ids,users);assert {k:len(v) for k,v in ix.items()}==cfg['counts']
 source=np.load(cfg['source_maps'],mmap_mode='r');assert source.shape==(6900,90,567) and source.dtype==np.float32
 raw=np.load(cache/'X_clutterremoved.npy',mmap_mode='r');subjects=np.load(cache/'subj_clutterremoved.npy',mmap_mode='r');np.testing.assert_array_equal(subjects[ids],users)
 for offset in range(0,6900,64): np.testing.assert_allclose(source[offset:offset+64],preprocess(np.array(raw[ids[offset:offset+64]],copy=True)),rtol=2e-5,atol=2e-6)
 labels=np.load(cache/'y_clutterremoved.npy',mmap_mode='r');yv=np.array(labels[ids[ix['validation']]],copy=True);yd=np.array(labels[ids[ix['diagnostic']]],copy=True)
 details=[];scores={};sys.path.insert(0,cfg['prototype']);from author_core import make_model
 for args in PHASES:
  report,best=verify_stage(base,args,cfg,identity,ids,ix,hashes);details.append(report)
  if args[1]=='pretrain': continue
  model,_=make_model(cfg['seed']);model.load_state_dict(best['model']);model.eval()
  def forward(index,targets=None):
   parts=[];loss_total=0.
   with torch.inference_mode():
    for offset in range(0,len(index),16):
     logits=model(torch.from_numpy(np.array(source[index[offset:offset+16]],copy=True)).transpose(1,2).contiguous());assert torch.isfinite(logits).all();parts.append(logits.numpy())
     if targets is not None:
      logprob=F.log_softmax(logits,dim=-1);target=torch.from_numpy(targets[offset:offset+16]).long()
      loss=(.9*(-logprob.gather(1,target[:,None]).squeeze(1))+.1*(-logprob.mean(-1))).mean();loss_total+=float(loss)*len(logits)
   return np.concatenate(parts),loss_total/len(index)
  validation,loss=forward(ix['validation'],yv);assert abs(metric(yv,validation.argmax(1))-best['validation_macro'])<1e-12
  np.testing.assert_allclose(loss,best['validation_loss'],rtol=1e-5,atol=1e-6)
  diagnostic,_=forward(ix['diagnostic']);arm=args[0]
  with np.load(run/arm/'predictions_unscored.npz') as z:
   np.testing.assert_array_equal(z['ids'],ids[ix['diagnostic']]);np.testing.assert_allclose(diagnostic,z['logits'],rtol=1e-5,atol=1e-6);np.testing.assert_array_equal(diagnostic.argmax(1),z['logits'].argmax(1));scores[arm]=metric(yd,z['logits'].argmax(1))
  report.update(validation_forward_records=len(yv),diagnostic_forward_records=len(yd),label_smoothing_CE_independently_recomputed=True,diagnostic_argmax_exact=True);del model,best
 prior=read(cfg['prior_held1_summary']);reference=max(prior['scores']['timemil'],prior['scores']['gap'],prior['cnn_reference']['full'],prior['cnn_reference']['local']);assert reference==cfg['reference_held1_macro']
 summary=read(run/'SUMMARY.json');assert summary['state']=='complete_exploratory_source_pilot' and summary['reference_held1_macro']==reference
 for arm in scores: assert abs(scores[arm]-summary['scores'][arm])<1e-12
 assert summary['gates']==decision(scores,reference) and summary['independent_validation'] is False and summary['novel_contribution_validated'] is False and summary['users7_8_labels_used'] is False
 assert summary['wall_seconds']<=43200 and summary['cpu_seconds']<=43200
 for p,h in hashes.items(): assert digest(p)==h,'changed during read-only audit: '+p
 result=dict(state='pass',scope='completed_TSLANet_pilot_read_only',phases=details,scores=scores,reference_held1_macro=reference,gates=decision(scores,reference),artifact_hashes=hashes,source_preprocessing_records_verified=6900,training_termination=term,auditor_sha256=digest(__file__),audit_wall_seconds=time.monotonic()-started,users7_8_labels_used=False,independent_validation=False,novel_contribution_validated=False,limitations='Both selected classifier validation/diagnostic forwards and independent smoothed CE replayed. SSL recorded masked-loss history/selection, states and transfer checked; original stochastic SSL validation loss NOT independently reconstructed (pre-validation RNG unlogged). Nonselected classifier/SSL epochs not retrained or replayed. RNG state reload validates schema, not entire stochastic history. No fitting or new method selection.')
 output.write_text(json.dumps(result,indent=2));print(json.dumps(dict(state='pass',output=str(output))))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--ssl-phase-only',action='store_true');args=parser.parse_args()
 (audit_ssl_phase if args.ssl_phase_only else audit)(BASE,args.out)
