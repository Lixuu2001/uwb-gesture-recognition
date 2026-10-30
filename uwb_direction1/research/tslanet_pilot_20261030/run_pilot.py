"""Fixed full-UEA TSLANet50SSL/100FT versus100scratch; no automatic launch."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import fcntl,hashlib,json,random,resource,signal,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/workspace/uwb-tslanet-screen-20261029')
from author_core import make_model
from training_ops import pretraining_loss,classification_criterion,stage_optimizer
B=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def write(p,value):
 p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(value,indent=2));q.replace(p)
def save(p,value):
 p=Path(p);q=p.with_suffix('.tmp');torch.save(value,q);q.replace(p)
def roles(users):return dict(train=np.flatnonzero(np.isin(users,[3,4,5,6])),validation=np.flatnonzero(users==2),diagnostic=np.flatnonzero(users==1))
def macro(y,p):return float(np.mean([np.mean(p[y==c]==c) for c in np.unique(y)]))
def gates(scores,reference):
 a=scores['pretrained'];b=scores['scratch']
 eps=1e-12
 mechanism=bool(a-b+eps>=.005 and a+eps>=reference-.005);scratch=bool(b+eps>=reference+.005);pretrained=bool(a+eps>=reference+.005)
 route='pretrained_mechanism' if mechanism else 'scratch_baseline' if scratch else 'pretrained_baseline' if pretrained else 'close_exact_configuration'
 return dict(mechanism_expansion_eligible=mechanism,scratch_baseline_expansion_eligible=scratch,pretrained_baseline_expansion_eligible=pretrained,predeclared_route=route)
def ready(cfg):
 state=json.loads(Path(cfg['previous_state']).read_text())
 if state.get('state')!='complete' or state.get('pid') is not None:raise RuntimeError('not_ready: prior GAP source6 still active/incomplete')
 for key in ('previous_audit','previous_private_backup'):
  value=json.loads(Path(cfg[key]).read_text());assert value['state']==('pass' if key=='previous_audit' else 'private_readback_verified'),key
 term=json.loads(Path(cfg['previous_termination']).read_text());assert term['exit_code']==0 and not term['wall_timeout']
def main():
 cfg=json.loads((B/'PROTOCOL.json').read_text());ready(cfg);assert cfg['cost_gate_passed']
 assert cfg['pretraining_epochs']==50 and cfg['classification_epochs_per_arm']==100 and cfg['batch_size']==16
 out=B/'run01';out.mkdir(exist_ok=False);lock=open(out/'worker.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 started=time.monotonic();state=dict(state='starting',pid=os.getpid(),completed_arms=[],diagnostic_labels_read=False,users7_8_labels_used=False)
 def record(**kw):
  state.update(kw);state.update(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time(),observed_unix=time.time());write(out/'STATE.json',state)
 def stop(sig,frame):raise TimeoutError('budget signal '+str(sig))
 for sig in (signal.SIGTERM,signal.SIGALRM,signal.SIGXCPU):signal.signal(sig,stop)
 resource.setrlimit(resource.RLIMIT_CPU,(43000,43100));signal.setitimer(signal.ITIMER_REAL,43100);record()
 try:
  frozen=json.loads((B/'FROZEN_HASHES.json').read_text())
  for n,h in frozen.items():assert sha(B/n)==h,n
  identity=dict(protocol_sha256=sha(B/'PROTOCOL.json'),frozen_hashes=frozen,python=sys.version,torch=torch.__version__,numpy=np.__version__);write(out/'IDENTITY.json',identity)
  previous=json.loads(Path(cfg['prior_held1_summary']).read_text());reference=max(previous['scores']['timemil'],previous['scores']['gap'],previous['cnn_reference']['full'],previous['cnn_reference']['local']);assert reference==cfg['reference_held1_macro']
  assert sha(cfg['source_manifest'])==cfg['source_manifest_sha256']
  with np.load(cfg['source_manifest']) as z:ids=z['ids'];users=z['users']
  assert len(ids)==len(np.unique(ids))==6900 and set(users)==set(range(1,7));ix=roles(users)
  assert {k:len(v) for k,v in ix.items()}==cfg['counts'];write(out/'split.json',{k:ids[v].tolist() for k,v in ix.items()})
  for n,h in cfg['input_sha256'].items():assert sha(Path(cfg['data'])/'cache'/n)==h
  assert sha(cfg['source_maps'])==cfg['source_maps_sha256']
  data=torch.from_numpy(np.array(np.load(cfg['source_maps'],mmap_mode='r'),copy=True)).transpose(1,2).contiguous();assert data.shape==(6900,567,90) and torch.isfinite(data).all()
  label_map=np.load(Path(cfg['data'])/'cache/y_clutterremoved.npy',mmap_mode='r');targets=None
  def classification_targets():
   nonlocal targets
   if targets is None:targets=(torch.from_numpy(np.array(label_map[ids[ix['train']]],copy=True)).long(),np.array(label_map[ids[ix['validation']]],copy=True))
   return targets
  def fresh_model():
   random.seed(cfg['seed']);np.random.seed(cfg['seed']);model,_=make_model(cfg['seed']);assert sum(p.numel() for p in model.parameters())==4132879;return model
  def phase(model,arm,kind,epochs,origin):
   folder=out/arm/kind;folder.mkdir(parents=True);opt=stage_optimizer(model);criterion=classification_criterion();best=None
   if kind=='classification':yt,yv=classification_targets()
   save(folder/'initial.pt',dict(model=model.state_dict(),identity=identity,epoch=0,arm=arm,phase=kind,origin=origin))
   for epoch in range(1,epochs+1):
    epstart=time.monotonic();model.train();order=torch.randperm(len(ix['train']),generator=torch.Generator().manual_seed(cfg['seed']+epoch)).numpy();used=(len(order)//16)*16;total=0.
    record(state='training',arm=arm,phase=kind,epoch=epoch,batch=0)
    for offset in range(0,used,16):
     j=order[offset:offset+16];opt.zero_grad(set_to_none=True);x=data[ix['train'][j]]
     loss=pretraining_loss(model,x) if kind=='pretrain' else criterion(model(x),yt[j]);assert torch.isfinite(loss);loss.backward()
     if offset==0:
      expected=set(cfg['unused_parameters'][kind]);assert {n for n,p in model.named_parameters() if p.grad is None}==expected
      assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
     opt.step();total+=float(loss.detach())*len(j)
     if not (folder/'first_update.pt').exists():save(folder/'first_update.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),training_ids=ids[ix['train'][j]].tolist(),identity=identity,arm=arm,phase=kind))
     if offset%(16*20)==0:record(batch=offset//16+1)
    model.eval();val_total=0.;val_logits=[]
    with torch.no_grad():
     for offset in range(0,len(ix['validation']),16):
      j=ix['validation'][offset:offset+16];x=data[j]
      if kind=='pretrain':vl=pretraining_loss(model,x)
      else:
       logits=model(x);assert torch.isfinite(logits).all();val_logits.append(logits);vl=criterion(logits,torch.from_numpy(yv[offset:offset+16]).long())
      assert torch.isfinite(vl);val_total+=float(vl)*len(j)
    val_loss=val_total/len(ix['validation']);vm=None if kind=='pretrain' else macro(yv,torch.cat(val_logits).argmax(1).numpy());key=(val_loss,epoch) if kind=='pretrain' else (-vm,val_loss,epoch)
    if best is None or key<best:best=key;save(folder/'best.pt',dict(model=model.state_dict(),identity=identity,epoch=epoch,validation_loss=val_loss,validation_macro=vm,arm=arm,phase=kind,origin=origin))
    save(folder/'last.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),identity=identity,epoch=epoch,best_selection=list(best),torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),arm=arm,phase=kind))
    row=dict(arm=arm,phase=kind,epoch=epoch,training_loss=total/used,training_records_used=used,validation_loss=val_loss,validation_macro=vm,epoch_seconds=time.monotonic()-epstart)
    with (folder/'epochs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    record(state='epoch_complete',**row);print(json.dumps(row),flush=True)
   checkpoint=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['model'])
   write(folder/'PHASE_COMPLETE.json',dict(arm=arm,phase=kind,epochs=epochs,selected_epoch=checkpoint['epoch'],checkpoint_sha256=sha(folder/'best.pt')));return model
  for arm in ('pretrained','scratch'):
   origin=dict(type='seeded_author_UEA_initialization',seed=cfg['seed'])
   if arm=='pretrained':
    model=phase(fresh_model(),arm,'pretrain',50,origin);del model
    checkpoint=out/arm/'pretrain/best.pt';origin=dict(type='selected_pretrain_model_only_fresh_optimizer_and_seed',path=str(checkpoint),sha256=sha(checkpoint));model=fresh_model();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'])
   else:model=fresh_model()
   model=phase(model,arm,'classification',100,origin);model.eval();parts=[]
   with torch.no_grad():
    for offset in range(0,len(ix['diagnostic']),16):parts.append(model(data[ix['diagnostic'][offset:offset+16]]))
   logits=torch.cat(parts);assert torch.isfinite(logits).all();np.savez_compressed(out/arm/'predictions_unscored.npz',ids=ids[ix['diagnostic']],logits=logits.numpy())
   write(out/arm/'ARM_COMPLETE.json',dict(arm=arm,classification_epochs=100,pretraining_epochs=50 if arm=='pretrained' else 0,prediction_sha256=sha(out/arm/'predictions_unscored.npz'),checkpoint_sha256=sha(out/arm/'classification/best.pt')))
   record(completed_arms=state['completed_arms']+[arm]);del model
  write(out/'ALL_PREDICTIONS_FROZEN.json',{arm:sha(out/arm/'predictions_unscored.npz') for arm in ('pretrained','scratch')});record(state='predictions_frozen')
  yd=np.array(label_map[ids[ix['diagnostic']]],copy=True);scores={}
  for arm in ('pretrained','scratch'):
   with np.load(out/arm/'predictions_unscored.npz') as z:np.testing.assert_array_equal(z['ids'],ids[ix['diagnostic']]);scores[arm]=macro(yd,z['logits'].argmax(1))
  assert reference==cfg['reference_held1_macro'];result=dict(state='complete_exploratory_source_pilot',scores=scores,reference_held1_macro=reference,gates=gates(scores,reference),independent_validation=False,novel_contribution_validated=False,users7_8_labels_used=False,wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time())
  write(out/'SUMMARY.json',result);record(state='complete',pid=None,diagnostic_labels_read=True);write(out/'FILES.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.suffix!='.tmp'});print(json.dumps(result),flush=True)
 except BaseException as e:
  record(state='incomplete',pid=None,error=repr(e),traceback=traceback.format_exc(),scientific_negative=False);raise
 finally:lock.close()
if __name__=='__main__':assert sys.argv[1:]==['--run'];main()
