"""Known MC100 estimator and exact immutable checkpoint/trace serialization."""
from pathlib import Path
import copy,hashlib,json,random,sys
sys.path.insert(0,'/workspace/uwb-mcdropout-source-probe-20261102')
from mc_dropout import np,torch,nn,infer,dropout_only
sys.path.insert(0,'/workspace/uwb-timemil-20261029')
from adapter import make_model
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def equal(a,b):
 if isinstance(a,torch.Tensor):assert torch.equal(a,b)
 elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
 elif isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(tuple,list)):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b
def state(m,identity):return copy.deepcopy(dict(model=m.state_dict(),modes={n:v.training for n,v in m.named_modules()},torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),identity=identity))
def model(seed,checkpoint=None):
 random.seed(seed);np.random.seed(seed);m=make_model('timemil',seed)
 if checkpoint is not None:
  s=torch.load(checkpoint,map_location='cpu',weights_only=False);assert s['epoch']==41;m.load_state_dict(s['model'],strict=True)
 m.eval();assert sum(p.numel() for p in m.parameters())==850973;return m
def deterministic(m,x,batch):
 m.eval();a=[]
 with torch.no_grad():
  for i in range(0,len(x),batch):a.append(m(torch.from_numpy(x[i:i+batch])).numpy())
 z=np.concatenate(a);assert z.shape==(len(x),12) and np.isfinite(z).all();return z
def produce(c,out,m,ids,ordered,maps,identity,record,reference=None):
 from artifacts import write,save
 initial=state(m,identity);save(out/'initial_state.pt',initial,torch);torch.save(m.state_dict(),out/'native_selected_weights.pt');frozen={};baseline={}
 for role in ['validation','diagnostic']:
  ix=ordered[role];x=np.array(maps[ix],dtype='float32',copy=True,order='C');pi=ids[ix];z=deterministic(m,x,c['batch_size'])
  if role=='diagnostic' and reference is not None:
   with np.load(reference) as ref:np.testing.assert_array_equal(ref['ids'],pi);np.testing.assert_array_equal(ref['logits'],z)
  baseline[role]=z;np.savez_compressed(out/(role+'_deterministic_unscored.npz'),ids=pi,logits=z,predictions=z.argmax(1))
  def after(t,logits,rng):
   row=dict(role=role,sample=t,sample_count=c['samples'],records=len(pi),ids_sha256=digest(pi),logits_sha256=digest(logits),torch_rng_sha256=digest(rng.numpy()))
   with (out/'passes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   if role=='validation' and t==1:save(out/'first_mc_pass.pt',dict(identity=identity,role=role,sample=t,ids=pi.copy(),logits=logits.copy(),torch_rng=rng.clone()),torch)
   record(role=role,samples_completed=t,total_forward_batches=t*((len(pi)+c['batch_size']-1)//c['batch_size']));print(json.dumps(row),flush=True)
  seed=c['seed']+(101 if role=='validation' else 102);r=infer(m,x,c['samples'],seed,c['batch_size'],after)
  assert len(r['dropout_modules'])==3 and all(d['p']==.2 for d in r['dropout_modules'])
  np.savez_compressed(out/(role+'_MC_unscored.npz'),ids=pi,logits=r['logits'],mean_sigmoid=r['mean_sigmoid'],predictions=r['predictions']);torch.save(torch.stack(r['torch_rng_after_each_sample']),out/(role+'_sample_rng.pt'));write(out/(role+'_COMPLETE.json'),dict(samples=c['samples'],records=len(pi),seed=seed,dropout_modules=r['dropout_modules']))
  frozen[role]=dict(MC=__import__('guards').sha(out/(role+'_MC_unscored.npz')),deterministic=__import__('guards').sha(out/(role+'_deterministic_unscored.npz')));equal(initial,state(m,identity))
 equal(initial,state(m,identity));save(out/'final_state.pt',state(m,identity),torch);write(out/'ALL_PREDICTIONS_FROZEN.json',frozen)
 return dict(state='complete_unscored_MC_inference',samples_per_role=c['samples'],baseline_same_ID_control_exact=reference is not None,deterministic_and_MC_both_roles_frozen=True,model_BN_buffers_and_global_RNG_unchanged=True,gradient_updates=0,diagnostic_labels_read=False,users7_8_labels_used=False)
def replay(c,out,m,ids,ordered,maps,identity,reference=None):
 initial=state(m,identity);equal(torch.load(out/'initial_state.pt',weights_only=False),initial);equal(torch.load(out/'native_selected_weights.pt',weights_only=True),m.state_dict());rows=(out/'passes.jsonl').open();total=0
 try:
  for role in ['validation','diagnostic']:
   ix=ordered[role];pi=ids[ix];x=np.array(maps[ix],dtype='float32',copy=True,order='C');z=deterministic(m,x,c['batch_size'])
   with np.load(out/(role+'_deterministic_unscored.npz')) as a:np.testing.assert_array_equal(a['ids'],pi);np.testing.assert_array_equal(a['logits'],z);np.testing.assert_array_equal(a['predictions'],z.argmax(1))
   if role=='diagnostic' and reference is not None:
    with np.load(reference) as a:np.testing.assert_array_equal(a['ids'],pi);np.testing.assert_array_equal(a['logits'],z)
   saved=np.load(out/(role+'_MC_unscored.npz'));np.testing.assert_array_equal(saved['ids'],pi);raw=saved['logits'];rngs=torch.load(out/(role+'_sample_rng.pt'),weights_only=True);assert raw.shape==(c['samples'],len(pi),12) and len(rngs)==c['samples'];sum_link=np.zeros((len(pi),12),dtype='float64');seed=c['seed']+(101 if role=='validation' else 102)
   # Deliberately explicit stochastic loop, independent of producer infer().
   with torch.random.fork_rng(devices=[],enabled=True),dropout_only(m),torch.no_grad():
    torch.manual_seed(seed)
    for t in range(c['samples']):
     chunks=[]
     for i in range(0,len(x),c['batch_size']):chunks.append(m(torch.from_numpy(x[i:i+c['batch_size']])).numpy())
     z=np.concatenate(chunks);np.testing.assert_array_equal(z,raw[t]);rng=torch.get_rng_state();assert torch.equal(rng,rngs[t]);sum_link+=torch.sigmoid(torch.from_numpy(z)).numpy().astype('float64');got=json.loads(next(rows));want=dict(role=role,sample=t+1,sample_count=c['samples'],records=len(pi),ids_sha256=digest(pi),logits_sha256=digest(z),torch_rng_sha256=digest(rng.numpy()));assert got==want;total+=1
     if role=='validation' and t==0:equal(torch.load(out/'first_mc_pass.pt',weights_only=False),dict(identity=identity,role=role,sample=1,ids=pi.copy(),logits=z.copy(),torch_rng=rng.clone()))
   # Same summation order as np.mean(axis=0,dtype=float64); all inputs exact.
   mean=sum_link/c['samples'];np.testing.assert_array_equal(mean,saved['mean_sigmoid']);np.testing.assert_array_equal(saved['predictions'],mean.argmax(1));saved.close();equal(initial,state(m,identity))
  assert rows.readline()=='';equal(torch.load(out/'final_state.pt',weights_only=False),state(m,identity));equal(initial,state(m,identity))
 finally:rows.close()
 return dict(samples_replayed=total,samples_per_role=c['samples'],all_stochastic_logits_and_rng_exact=True,independent_explicit_loop_and_float64_mean_exact=True,initial_first_final_states_and_native_selected_weights_exact=True,both_deterministic_controls_exact=True,model_BN_buffers_and_global_RNG_immutable=True,diagnostic_labels_read=False,gradient_updates=0)
