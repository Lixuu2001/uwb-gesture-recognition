"""Completed-run reconstruction of all5760updates; no candidate fitting/selection."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def require_complete(base):
 out=base/'run01';s=read(out/'STATE.json')
 if s['state']!='complete' or s['pid'] is not None:raise RuntimeError('not_ready: active/incomplete experiment')
 assert s['completed_phases']==['ssl','linear','finetune','scratch'] and s['all_predictions_frozen'] and s['diagnostic_labels_read'] and not s['users7_8_labels_used']
 term=read(base/'supervisor01/TERMINATION.json');assert term['exit_code']==0 and not term['wall_timeout'] and term['supervisor_wall_seconds']<=43200 and term['child_cpu_seconds']<=43200
 preds=read(out/'ALL_PREDICTIONS_FROZEN.json');assert len(preds)==6
 for key,h in preds.items():
  phase,role=key.split('/');assert sha(out/phase/(role+'_predictions_unscored.npz'))==h
 for phase in ('ssl','linear','finetune','scratch'):
  done=read(out/phase/'PHASE_COMPLETE.json');assert done['epochs']==40 and done['optimizer_updates']==1440 and done['checkpoint']==sha(out/phase/'last.pt')
 return term
def score(y,p):return float(sum(float(np.mean(p[y==c]==c)) for c in np.unique(y))/len(np.unique(y)))
def audit(output):
 term=require_complete(B);assert not output.exists();start=time.monotonic();cfg=read(B/'PROTOCOL.json');out=B/'run01';hashes={}
 def checked(p,h):assert sha(p)==h,str(p);hashes[str(p)]=h
 for n,h in read(B/'FROZEN_HASHES.json').items():checked(B/n,h)
 for n,h in read(out/'FILES.json').items():checked(out/n,h)
 import torch,warnings
 from ops import make,step,indices,augmented,save_state,equal
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);warnings.filterwarnings('ignore',message='Implicit dimension choice')
 identity=read(out/'IDENTITY.json');assert identity['python']==sys.version and identity['torch']==torch.__version__ and identity['numpy']==np.__version__ and identity['protocol_sha256']==sha(B/'PROTOCOL.json') and identity['frozen_hashes']==read(B/'FROZEN_HASHES.json')
 checked(cfg['source_manifest'],cfg['source_manifest_sha256']);checked(cfg['source_maps'],cfg['source_maps_sha256']);cache=Path(cfg['data'])/'cache'
 for n,h in cfg['input_sha256'].items():checked(cache/n,h)
 with np.load(cfg['source_manifest']) as z:ids=z['ids'];users=z['users']
 assert len(ids)==len(np.unique(ids))==6900 and set(users)==set(range(1,7));ix=dict(train=np.flatnonzero(np.isin(users,[3,4,5,6])),validation=np.flatnonzero(users==2),diagnostic=np.flatnonzero(users==1));split=read(out/'split.json')
 for role,index in ix.items():np.testing.assert_array_equal(split[role],ids[index]);assert len(index)==cfg['counts'][role]
 subjects=np.load(cache/'subj_clutterremoved.npy',mmap_mode='r');np.testing.assert_array_equal(subjects[ids],users)
 maps=np.load(cfg['source_maps'],mmap_mode='r');raw=np.load(cache/'X_clutterremoved.npy',mmap_mode='r')
 for start0 in range(0,6900,64):
  x=np.array(raw[ids[start0:start0+64]],copy=True);rms=np.sqrt(np.mean(x*x,axis=(2,3),keepdims=True,dtype=np.float32));ref=(x/np.maximum(rms,np.float32(1e-8))).transpose(0,2,1,3).reshape(len(x),90,567);np.testing.assert_allclose(ref,maps[start0:start0+64],rtol=2e-5,atol=2e-6)
 train=np.array(maps[ix['train']].transpose(0,2,1),copy=True,order='C');weak,strong,recipe=augmented(train,cfg['seed']);equal(recipe,torch.load(out/'augmentation_recipe.pt',map_location='cpu',weights_only=False));aug=read(out/'AUGMENTATION.json');assert hashlib.sha256(weak).hexdigest()==aug['weak_sha256'] and hashlib.sha256(strong).hexdigest()==aug['strong_sha256']
 labels=np.load(cache/'y_clutterremoved.npy',mmap_mode='r');yt=np.array(labels[ids[ix['train']]],copy=True);yv=np.array(labels[ids[ix['validation']]],copy=True);validation_values=np.array(maps[ix['validation']].transpose(0,2,1),copy=True,order='C');pretrained=None;scores={};vals={};replayed=0
 def forward(m,values):
  m.eval();pieces=[]
  with torch.inference_mode():
   for i in range(0,len(values),128):pieces.append(m(torch.from_numpy(np.array(values[i:i+128],copy=True)))[0].numpy())
  return np.concatenate(pieces)
 for phase in ('ssl','linear','finetune','scratch'):
  stage=out/phase;c,m,t,o,to,s=make(cfg['seed'],phase,pretrained);rows=[json.loads(x) for x in (stage/'iterations.jsonl').read_text().splitlines()];epochs=[json.loads(x) for x in (stage/'epochs.jsonl').read_text().splitlines()];assert len(rows)==1440 and len(epochs)==40
  def compare(name,epoch,batch,total):
   saved=torch.load(stage/name,map_location='cpu',weights_only=False);expected=save_state(m,t,o,to,s,phase,epoch,batch,total,identity)
   if name=='first_update.pt':np.testing.assert_array_equal(saved.pop('training_ids'),ids[ix['train'][indices(1,0,cfg['seed'])]])
   equal(saved,expected)
  compare('initial.pt',0,-1,0)
  for epoch in range(1,41):
   losses=[]
   for batch in range(36):
    ii=indices(epoch,batch,cfg['seed']);total=(epoch-1)*36+batch+1;row=rows[total-1];assert row['epoch']==epoch and row['batch']==batch and row['total_steps']==total;np.testing.assert_array_equal(row['training_ids'],ids[ix['train'][ii]])
    y=np.zeros(128,dtype=np.int64) if phase=='ssl' else yt[ii]
    loss,_=step(m,t,o,to,c,torch.from_numpy(train[ii]),torch.from_numpy(y),torch.from_numpy(weak[ii]) if phase=='ssl' else None,torch.from_numpy(strong[ii]) if phase=='ssl' else None,phase);assert float(loss)==row['loss'];losses.append(float(loss));replayed+=1
    if total==1:compare('first_update.pt',1,0,1)
   m.eval();t.eval();record=epochs[epoch-1];assert record['epoch']==epoch and record['steps']==epoch*36 and abs(record['training_loss']-float(np.mean(losses)))<=1e-12
   if phase!='ssl':
    logits=forward(m,validation_values);ce=[float(torch.nn.functional.cross_entropy(torch.from_numpy(logits[i:i+128]),torch.from_numpy(yv[i:i+128]).long())) for i in range(0,len(yv),128)];vl=float(np.mean(ce));assert abs(vl-record['validation_loss'])<=1e-12
    shift=logits.astype(np.float64)-logits.max(1,keepdims=True);nll=np.log(np.exp(shift).sum(1))-shift[np.arange(len(yv)),yv];independent=float(np.mean([nll[i:i+128].mean() for i in range(0,len(yv),128)]));assert abs(independent-vl)<2e-5
    assert abs(score(yv,logits.argmax(1))-record['validation_macro'])<=1e-12;s.step(vl);assert o.param_groups[0]['lr']==record['learning_rate']
   if epoch%10==0:compare(f'epoch{epoch:02d}.pt',epoch,35,epoch*36)
  compare('last.pt',40,35,1440)
  assert {int(v['step']) for v in o.state.values()}=={1440} and len(o.state)==({'ssl':9,'linear':2,'finetune':11,'scratch':11}[phase])
  if phase=='ssl':
   assert len(to.state)==65 and {int(v['step']) for v in to.state.values()}=={1440};pretrained={k:v.detach().clone() for k,v in m.state_dict().items()}
  else:
   assert len(to.state)==0
   for role,container in [('validation',vals),('diagnostic',scores)]:
    values=validation_values if role=='validation' else np.array(maps[ix[role]].transpose(0,2,1),copy=True,order='C');logits=forward(m,values)
    with np.load(stage/(role+'_predictions_unscored.npz')) as z:np.testing.assert_array_equal(z['ids'],ids[ix[role]]);np.testing.assert_array_equal(z['logits'],logits);np.testing.assert_array_equal(z['predictions'],logits.argmax(1))
    container[phase]=score(np.array(labels[ids[ix[role]]],copy=True),logits.argmax(1))
 summary=read(out/'SUMMARY.json');assert replayed==5760
 for phase in scores:assert abs(scores[phase]-summary['scores'][phase])<=1e-12 and abs(vals[phase]-summary['validation_scores'][phase])<=1e-12
 eligible=any(v-cfg['reference_held1_macro']+1e-12>=.005 for v in scores.values());assert eligible==summary['known_baseline_expansion_eligible'] and summary['predeclared_route']==('known_baseline_source_confirmation' if eligible else 'close_exact_configuration')
 for p,h in hashes.items():assert sha(p)==h,p
 result=dict(state='pass',scores=scores,producer_scores=summary['scores'],reference_held1_macro=cfg['reference_held1_macro'],known_baseline_expansion_eligible=eligible,training_updates_replayed_exact=replayed,training_appearances_verified=737280,all_four_phase_initial_first_every10_last_models_optimizers_schedulers_RNG_bitwise=True,all6900RMS_inputs_verified=True,fixed4700augmentation_views_and_recipe_exact=True,validation_and_diagnostic_logits_verified=6600,all120validation_epochs_replayed=True,artifact_hashes=hashes,training_termination=term,auditor_sha256=sha(__file__),audit_wall_seconds=time.monotonic()-start,users7_8_labels_used=False,independent_validation=False,novel_contribution_validated=False,limitations='Exploratory user-disjoint adaptation, documented augmentation/axis/RNG-order differences; no original architecture or independent test claim. Macro reduction tolerance1e-12; all tensors/optimizers/logits/RNG exact.')
 output.write_text(json.dumps(result,indent=2));print(json.dumps(dict(state='pass',output=str(output),updates=replayed)))
if __name__=='__main__':assert sys.argv[1:]==['--run'];audit(B/'COMPLETE_AUDIT.json')
