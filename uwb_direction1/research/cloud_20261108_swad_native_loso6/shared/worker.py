"""Full native dense SWAD64/fresh64 and source-only controls; never held truth."""
from pathlib import Path
import copy,fcntl,hashlib,json,sys,time,shutil
from common import B,ready,read,write,sha
import engine as e
from data import load
from journal import StepLog,commit,load as journal_load

def row_for(ids,loc,pos,epoch,offset,updates,step):
 return dict(epoch=epoch,offset=offset,update=updates,positions_sha256=hashlib.sha256(pos.tobytes()).hexdigest(),training_ids_sha256=hashlib.sha256(ids[loc['train'][pos]].tobytes()).hexdigest(),**step)
def baseline_first(c,actual,ids,loc,pos):
 if c['generated_fixture']:return dict(state='generated_fixture_no_real_GAP_checkpoint_read')
 r=c['baseline_first'];old=e.legacy_checkpoint(r['path'],r['sha256']);assert e.semantic(old['model'])==e.semantic(actual['model']) and e.semantic(old['optimizer'])==e.semantic(actual['optimizer']['native_optimizer'])
 e.np.testing.assert_array_equal(e.np.asarray(old['training_ids']),ids[loc['train'][pos]])
 return dict(state='original_GAP_first_model_optimizer_and_source_IDs_exact',legacy_TorchVersion_only_compatibility=True,original_file_sha256=r['sha256'])
def original_curve(c):
 if c['generated_fixture']:return None
 p=c['baseline_curve'];assert sha(p['path'])==p['sha256'];rows=[json.loads(x) for x in Path(p['path']).read_text().splitlines()];assert len(rows)==64
 return rows
def check_curve(rows,epoch,vm,vl,train_bce):
 if rows is None:return
 r=rows[epoch-1];assert r['epoch']==epoch
 assert vm==r['validation_macro'] and vl==r.get('validation_bce',r.get('validation_loss')),'Native validation trajectory differs from original GAP'
 assert train_bce==r['train_bce'],'Native training loss trajectory differs from original GAP'
def baseline_selected(c,arm,ids,loc):
 if c['generated_fixture']:return dict(state='generated_fixture_no_real_GAP_selected_checkpoint_read')
 ref=c['baseline_artifacts'];original=e.legacy_checkpoint(ref['model'],ref['model_sha256'])
 assert e.semantic(original['model'])==e.semantic(arm['model']),'Native selected model differs from original strong GAP'
 assert original['epoch']==arm['metadata']['selection']['native_reference_selected_epoch']
 assert sha(ref['prediction'])==ref['prediction_sha256']
 with e.np.load(ref['prediction']) as z:
  e.np.testing.assert_array_equal(z['ids'],ids[loc['diagnostic']]);e.np.testing.assert_array_equal(z['logits'],arm['scores']['diagnostic'])
 return dict(state='original_GAP_selected_model_epoch_and_unscored_outputs_exact',model_sha256=ref['model_sha256'],prediction_sha256=ref['prediction_sha256'])
def first():
 c=ready(True);ids,u,loc,maps,y,v=load(True);m,o,g=e.make();st=e.torch.load(B/'run/train/initial.pt',map_location='cpu',weights_only=True)
 e.restore(st,m,o,g);m.train();pos=e.order(len(y),1)[:64]
 result=e.step(m,o,e.np.array(maps[loc['train'][pos]],copy=True),y[pos],1,u[loc['train'][pos]])
 actual=copy.deepcopy(e.capture(m,o,g,0,1));grads=[p.grad.detach().clone() for p in m.parameters()];stored=e.torch.load(B/'run/train/first_update.pt',map_location='cpu',weights_only=True)
 assert e.semantic(actual)==e.semantic(stored) and read(B/'run/train/FIRST_READY.json')['step']==row_for(ids,loc,pos,1,0,1,result)
 original=baseline_first(c,actual,ids,loc,pos)
 for name,p in m.named_parameters():e.torch.testing.assert_close(actual['optimizer']['swad']['segment']['model'][name],p.detach(),rtol=0,atol=0)
 e.restore(st,m,o,g);m.train();z=m(e.torch.from_numpy(e.np.array(maps[loc['train'][pos]],copy=True)));zz=z.detach().numpy().astype('float64');hot=e.np.eye(12)[y[pos]]
 ref_loss=float((e.np.logaddexp(0,zz)-hot*zz).mean());gradient=e.torch.as_tensor((1/(1+e.np.exp(-zz))-hot)/(len(pos)*12),dtype=z.dtype)
 o.zero_grad();reference=e.torch.autograd.grad(z,tuple(m.parameters()),grad_outputs=gradient)
 for p,a in zip(m.parameters(),reference):p.grad=a
 e.torch.nn.utils.clip_grad_norm_(m.parameters(),2.);diff=[]
 for a,p in zip(grads,m.parameters()):e.torch.testing.assert_close(a,p.grad,rtol=2e-4,atol=2e-6);diff.append(float((a-p.grad).abs().max()))
 assert abs(ref_loss-result['loss'])<=2e-6
 write(B/'run/train/FIRST_INDEPENDENT_REFERENCE.json',dict(state='full617612_native_globalBCE_first_gradient_NumPy_reference_and_original_GAP_first_state_verified',max_gradient_absolute_difference=max(diff),reference_loss=ref_loss,parameters=617612,native_reference=original,segment_first_parameters_equal_native_post_update=True,method_state_native_replay_bitwise=True,diagnostic_labels_read=False,does_not_exclude_all_shared_implementation_errors=True))
 write(B/'run/train/FIRST_RELEASE.json',dict(first_sha256=sha(B/'run/train/first_update.pt'),native_first_update_exact=True,state_sha256=e.semantic(actual),diagnostic_labels_read=False))
def save_outputs(c,out,m,o,g,ids,u,loc,maps,state,replay=False):
 allarms,cal_cost=e.make_outputs(m,o,ids,u,loc,maps,state['best_model'],state['best_epoch']);native_match=baseline_selected(c,allarms['native_reference'],ids,loc)
 for arm,data in allarms.items():
  selected=dict(model=data['model'],metadata=data['metadata']);e.atomic_save(out/f'selected_{arm}.pt',selected)
  for role,z in data['scores'].items():
   e.np.savez_compressed(out/f'{arm}_{role}_unscored.npz',ids=ids[loc[role]],scores=z)
   if replay:
    old=B/f'run/train/{arm}_{role}_unscored.npz'
    with e.np.load(old) as p:e.np.testing.assert_array_equal(p['ids'],ids[loc[role]]);e.np.testing.assert_array_equal(p['scores'],z)
  if replay:
   old=e.torch.load(B/f'run/train/selected_{arm}.pt',map_location='cpu',weights_only=True);assert e.semantic(old)==e.semantic(selected)
 shutil.copyfile(out/'selected_swad.pt',out/'selected.pt')
 for role in ['validation','diagnostic']:shutil.copyfile(out/f'swad_{role}_unscored.npz',out/f'{role}_unscored.npz')
 write(out/'OUTPUT_SELECTION.json',dict(swad=allarms['swad']['metadata'],tail=allarms['tail']['metadata'],bn_only=allarms['bn_only']['metadata'],native_reference=native_match))
 write(out/'CALIBRATION_COST.json',cal_cost)
 write(out/'EVALUATION_BATCH_COST.json',e.EVAL_BATCH_COST)
 return allarms

def fit(replay=False,resume=False):
 c=ready(True);root=B/'run';out=root/('replay' if replay else 'train')
 if resume:assert out.exists()
 else:out.mkdir(parents=True,exist_ok=False)
 lock=(out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(out/'COMPLETE.json').exists()
 m,o,g=e.make();ids,u,loc,maps,y,v=load(True);curves=original_curve(c)
 if not(out/'split.npz').exists():e.np.savez_compressed(out/'split.npz',**{k:ids[p] for k,p in loc.items()})
 with e.np.load(out/'split.npz') as z:
  for k,p in loc.items():e.np.testing.assert_array_equal(z[k],ids[p])
 state,p=journal_load(out);checkpoint_cost=[]
 if state is None:
  state=e.capture(m,o,g,0,0);initial=e.torch.load(B/'initial.pt',map_location='cpu',weights_only=True);assert e.semantic(state)==e.semantic(initial)
  e.atomic_save(out/'initial.pt',state);log=StepLog(out/'steps.jsonl');rec=e.record(state,0,0)
  if replay:assert rec==read(root/'train/epoch000.json')
  wt=time.monotonic();ct=time.process_time();commit(out,state,rec,log.prefix());checkpoint_cost.append(dict(epoch=0,wall=time.monotonic()-wt,CPU=time.process_time()-ct))
 else:
  assert resume;e.restore(state,m,o,g);log=StepLog(out/'steps.jsonl',state['updates'],p['prefix_bytes'],p['prefix_sha256'])
 reference=(root/'train/steps.jsonl').open() if replay else None
 if replay:
  for _ in range(state['updates']):assert reference.readline()
 updates=state['updates'];best_key=state['best_key'];best_model=state['best_model'];best_epoch=state['best_epoch']
 try:
  for epoch in range(state['epoch']+1,65):
   m.train();order=e.order(len(y),epoch);total=0.
   for offset in range(0,len(y),64):
    pos=order[offset:offset+64];step=e.step(m,o,e.np.array(maps[loc['train'][pos]],copy=True),y[pos],epoch,u[loc['train'][pos]]);updates+=1;total+=step['loss']*len(pos)
    row=row_for(ids,loc,pos,epoch,offset,updates,step)
    if replay:assert json.loads(reference.readline())==row,'Full fresh replay step mismatch'
    log.append(row)
    if updates==1 and not replay:
     e.atomic_save(out/'first_update.pt',e.capture(m,o,g,0,1));write(out/'FIRST_READY.json',dict(step=row,checkpoint_sha256=sha(out/'first_update.pt')));start=time.monotonic()
     while not(out/'FIRST_RELEASE.json').exists():assert time.monotonic()-start<600,'First audit timeout';time.sleep(.2)
     release=read(out/'FIRST_RELEASE.json');assert release['first_sha256']==sha(out/'first_update.pt') and release['native_first_update_exact']
   scores,vm,vl=e.evaluate(m,maps,loc['validation'],v);check_curve(curves,epoch,vm,vl,total/len(y));key=(vm,-vl,-epoch)
   if best_key is None or key>tuple(best_key):best_key=key;best_epoch=epoch;best_model=copy.deepcopy(m.state_dict())
   o.epoch_end(epoch,vm,vl)
   wt=time.monotonic();ct=time.process_time();state=e.capture(m,o,g,epoch,updates,best_key,best_model,best_epoch);rec=e.record(state,epoch,updates,vm,vl,scores)
   if replay:assert rec==read(root/f'train/epoch{epoch:03}.json'),'Full fresh replay epoch/native/method state mismatch'
   commit(out,state,rec,log.prefix());checkpoint_cost.append(dict(epoch=epoch,wall=time.monotonic()-wt,CPU=time.process_time()-ct))
   write(out/'STATE.json',dict(state='full_replay' if replay else 'training',epoch=epoch,updates=updates,native_reference_selected_epoch=best_epoch,swad_converge_step=o.swad.valley.converge_step,swad_dead_valley=o.swad.valley.dead_valley,diagnostic_labels_read=False,users7_8_labels_used=False))
   print(json.dumps(dict(stage='replay' if replay else 'train',epoch=epoch,updates=updates,validation_macro=vm)),flush=True)
  assert updates==c['expected_updates'] and log.index>=log.original
  if replay:assert reference.read()==''
  shutil.copyfile(out/'slot0.pt',out/'final_full.pt');terminal=e.semantic(state)
  allarms=save_outputs(c,out,m,o,g,ids,u,loc,maps,state,replay)
  assert e.semantic(e.capture(m,o,g,64,updates,best_key,best_model,best_epoch))==terminal
  write(out/'CHECKPOINT_COST.json',checkpoint_cost);write(out/'STEP_DIGEST_COST.json',e.STEP_DIGEST_COST)
  write(out/'COMPLETE.json',dict(state='full64_fresh_replay_exact_all_arms_unscored' if replay else 'full64_dense_native_training_and_all_arms_complete_unscored',epochs=64,updates=updates,native_reference_selected_epoch=best_epoch,SWAD_selection=allarms['swad']['metadata']['selection'],final_state_sha256=terminal,selected_sha256=sha(out/'selected.pt'),validation_outputs_sha256=sha(out/'validation_unscored.npz'),diagnostic_outputs_sha256=sha(out/'diagnostic_unscored.npz'),arms=c['arms'],native_original_curve_and_selected_reference_verified=not c['generated_fixture'],diagnostic_labels_read=False,users7_8_labels_used=False))
 finally:
  log.close();lock.close()
  if reference:reference.close()
def output_audit():
 c=ready(True);ids,u,loc,maps,y,v=load(True);m,o,g=e.make();root=B/'run';st,_=journal_load(root/'train');fresh,_=journal_load(root/'replay');assert e.semantic(st)==e.semantic(fresh)
 e.restore(st,m,o,g);arms,calcost=e.make_outputs(m,o,ids,u,loc,maps,st['best_model'],st['best_epoch']);baseline_selected(c,arms['native_reference'],ids,loc)
 for arm,data in arms.items():
  saved=e.torch.load(root/f'train/selected_{arm}.pt',map_location='cpu',weights_only=True);assert e.semantic(saved)==e.semantic(dict(model=data['model'],metadata=data['metadata']))
  for role,z in data['scores'].items():
   with e.np.load(root/f'train/{arm}_{role}_unscored.npz') as old:e.np.testing.assert_array_equal(old['ids'],ids[loc[role]]);e.np.testing.assert_array_equal(old['scores'],z)
 files={}
 for side in ['train','replay']:
  for arm in c['arms']:
   for n in [f'selected_{arm}.pt',f'{arm}_validation_unscored.npz',f'{arm}_diagnostic_unscored.npz']:files[f'{side}/{n}']=sha(root/side/n)
 for n in ['train/selected.pt','train/final_full.pt','train/validation_unscored.npz','train/diagnostic_unscored.npz','replay/COMPLETE.json']:files[n]=sha(root/n)
 write(root/'OUTPUT_AUDIT_CALIBRATION_COST.json',calcost);write(root/'OUTPUT_AUDIT_EVAL_COST.json',e.EVAL_BATCH_COST)
 frozen=dict(state='all4_arms_native64_fresh64_BN_calibrated_outputs_verified_unscored',epochs=64,updates=c['expected_updates'],native_reference_selected_epoch=st['best_epoch'],SWAD_selection=arms['swad']['metadata']['selection'],full_training_replayed=True,native_reference_matches_original_GAP=not c['generated_fixture'],diagnostic_labels_read=False,users7_8_labels_used=False,files=files)
 write(root/'PREDICTIONS_FROZEN.json',frozen);print(json.dumps(frozen))
if __name__=='__main__':
 assert sys.argv[1] in ['train','replay','first','output']
 if sys.argv[1] in ['train','replay']:fit(sys.argv[1]=='replay','--resume' in sys.argv[2:])
 elif sys.argv[1]=='first':first()
 else:output_audit()
