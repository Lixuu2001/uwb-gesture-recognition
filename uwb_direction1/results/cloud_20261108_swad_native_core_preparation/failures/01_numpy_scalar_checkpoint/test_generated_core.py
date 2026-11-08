"""Generated-only SWAD wrapper/native dense averaging/checkpoint checks, one fixed profile."""
from pathlib import Path
import sys,os,copy,json,hashlib,random,time,tempfile,math
import numpy as np
import torch
P=Path(__file__).resolve().parent
# Refuse actual UWB .npy/.npz and raw restored-directory value opens.
def firewall(event,args):
 if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
 path=os.fsdecode(args[0])
 if path.startswith('/workspace/uwb-restored-') or (path.startswith('/workspace/') and path.endswith(('.npy','.npz')) and not path.startswith(str(P)+'/')):raise AssertionError('Generated-only firewall blocked real input value access')
sys.addaudithook(firewall)
import swad_core as c
sys.path.insert(0,'/workspace/uwb-timemil-20261029');import adapter as native
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
def semantic(value):
 h=hashlib.sha256()
 def walk(v):
  if isinstance(v,torch.Tensor):h.update(str((tuple(v.shape),str(v.dtype))).encode());h.update(v.detach().contiguous().cpu().numpy().tobytes())
  elif isinstance(v,dict):
   for k,x in v.items():h.update(str(k).encode());walk(x)
  elif isinstance(v,(list,tuple)):
   for x in v:walk(x)
  else:h.update(str(v).encode())
 walk(value);return h.hexdigest()
class Toy(torch.nn.Module):
 def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.tensor([0.],dtype=torch.float64));self.register_buffer('marker',torch.tensor([9.],dtype=torch.float64))
def toy_cases():
 # Independently worked exact source corner cases. Original code can include repeated
 # smooth-queue segments and leaves buffers at the constructor state; do not "fix" it.
 cases=[([3.,2.,1.],31.,13.,2),([1.,2.,3.],21.,9.,1),([.1,.1,.1],26.,9.,4)]
 rows=[]
 for losses,expected,marker,n in cases:
  m=Toy();s=c.NativeSWADState(m);updates=0
  for epoch,loss in enumerate(losses,1):
   for val in [epoch*10.,epoch*10.+2.]:
    with torch.no_grad():m.p.fill_(val);m.marker.add_(1)
    m.train();updates+=1;s.post_step(m,updates,epoch)
   m.eval();s.epoch_end(m,epoch,.5,loss)
  a=s.finalize();assert a.module.p.item()==expected and a.module.marker.item()==marker and a.n_averaged.item()==n
  try:s.finalize()
  except AssertionError:pass
  else:raise AssertionError('Double finalization allowed')
  rows.append(dict(losses=losses,value=expected,constructor_buffer=marker,final_n_averaged=n))
 # Actual 64-epoch scalar fixture checks tail33..64 inclusion without neural timing.
 m=Toy();s=c.NativeSWADState(m)
 for epoch in range(1,65):
  with torch.no_grad():m.p.fill_(epoch)
  m.train();s.post_step(m,epoch,epoch);m.eval();s.epoch_end(m,epoch,.5,.1 if epoch<=3 else 2.)
 assert s.tail.n_averaged.item()==32 and s.tail.start_step==33 and s.tail.end_step==64 and s.tail.module.p.item()==48.5
 assert s.valley.dead_valley
 # Reject invalid validation before touching queue state.
 before=semantic(s.state_dict());m.train();s.post_step(m,65,65);before=semantic(s.state_dict())
 try:s.epoch_end(m,65,.5,float('nan'))
 except AssertionError:pass
 else:raise AssertionError('NaN validation accepted')
 assert semantic(s.state_dict())==before
 return rows

def make():
 random.seed(20261029);np.random.seed(20261029);m=native.make_model('gap',20261029)
 assert sum(p.numel() for p in m.parameters())==617612
 return m,native.StableLookahead(m.parameters(),lr=.001,weight_decay=.0001),torch.Generator().manual_seed(20261029)
def step(m,o,x,y):
 m.train();z=m(x);loss=torch.nn.functional.binary_cross_entropy_with_logits(z,torch.nn.functional.one_hot(y,12).float());o.zero_grad();loss.backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters());torch.nn.utils.clip_grad_norm_(m.parameters(),2.);o.step();return float(loss.detach())
def capture(m,o,g,s):
 n=np.random.get_state()
 return copy.deepcopy(dict(model=m.state_dict(),optimizer=o.state_dict(),order_rng=g.get_state(),torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=(n[0],torch.from_numpy(n[1].copy()),n[2],n[3],n[4]),module_modes={k:a.training for k,a in m.named_modules()},swad=None if s is None else s.state_dict()))
def restore(st,m,o,g,s=None):
 m.load_state_dict(st['model'],strict=True);o.load_state_dict(copy.deepcopy(st['optimizer']));g.set_state(st['order_rng']);torch.set_rng_state(st['torch_rng']);random.setstate(st['python_rng']);n=st['numpy_rng'];np.random.set_state((n[0],n[1].numpy(),n[2],n[3],n[4]))
 for k,a in m.named_modules():a.training=st['module_modes'][k]
 if s is not None:s.load_state_dict(st['swad'],m)
def native_checks():
 m,o,g=make();s=c.NativeSWADState(m);initial=capture(m,o,g,s);dense=c.CPUAverage(m);initial_buffers={k:v.clone() for k,v in dense.module.named_buffers()};means={};native_rows=[];snapshot=None
 x=torch.from_numpy(np.random.default_rng(155).normal(size=(8,90,567)).astype('float32'));y=torch.arange(8)%12
 maxdiff=0.
 for j in range(1,9):
  loss=step(m,o,x,y);s.post_step(m,j,(j-1)//2+1);dense.update_parameters(m,step=j)
  for k,p in m.named_parameters():
   a=p.detach().numpy().astype('float64');means[k]=a if j==1 else means[k]+(a-means[k])/j
  if j%2==0:m.eval();s.epoch_end(m,j//2,.5,[.1,.1,.1,2.][j//2-1])
  native_rows.append(semantic(dict(model=m.state_dict(),optimizer=o.state_dict(),torch_rng=torch.get_rng_state(),mode=m.training)))
  if j==7:snapshot=capture(m,o,g,s)
  if j==8:expected=capture(m,o,g,s)
 for k,p in dense.module.named_parameters():maxdiff=max(maxdiff,float(abs(p.detach().numpy().astype('float64')-means[k]).max()))
 assert maxdiff<=2e-6
 assert all(torch.equal(v,dict(dense.module.named_buffers())[k]) for k,v in initial_buffers.items())
 assert o.count==8 and o.slow
 with tempfile.TemporaryDirectory(dir=P) as folder:
  path=Path(folder)/'full_native_swad.pt';torch.save(snapshot,path);saved=torch.load(path,map_location='cpu',weights_only=True)
  restore(saved,m,o,g,s);step(m,o,x,y);s.post_step(m,8,4);m.eval();s.epoch_end(m,4,.5,2.)
  assert semantic(capture(m,o,g,s))==semantic(expected)
  final_a=s.finalize();final_digest=semantic(c._pack(final_a))
  restore(expected,m,o,g,s);final_b=s.finalize();assert semantic(c._pack(final_b))==final_digest
 # Remove all averaging calls and replay the same eight native updates; model,
 # native optimizer/Lookahead and Torch RNG must remain literally unchanged.
 restore(initial,m,o,g)
 for j in range(1,9):
  step(m,o,x,y)
  if j%2==0:m.eval()
  assert semantic(dict(model=m.state_dict(),optimizer=o.state_dict(),torch_rng=torch.get_rng_state(),mode=m.training))==native_rows[j-1]
 return dict(parameters=617612,independent_float64_dense_parameter_mean_maxdiff=maxdiff,native_trajectory_optimizer_all8_RNG_bitwise_unchanged_by_collecting_average=True,slow6_disk7_exact8_full_queues_average_RNG_mode=True,finalize_restoration_exact=True,buffers_not_averaged_constructor_retained=True)

def profile():
 plan=json.loads((P/'PROTOCOL.json').read_text())['profile_before_measure'];assert plan['updates_per_arm']==8 and plan['warmup']==3
 models={}
 for arm in ['native','SWAD']:
  m,o,g=make();models[arm]=(m,o,None if arm=='native' else c.NativeSWADState(m))
 x=torch.from_numpy(np.random.default_rng(471).normal(size=(64,90,567)).astype('float32'));y=torch.arange(64)%12;times={a:[] for a in models}
 for j in range(1,9):
  for arm in (['native','SWAD'] if j%2 else ['SWAD','native']):
   m,o,s=models[arm];wall=time.monotonic();cpu=time.process_time();step(m,o,x,y)
   if s:s.post_step(m,j,1)
   times[arm].append(dict(wall=time.monotonic()-wall,CPU=time.process_time()-cpu))
 ratios={unit:max(a[unit] for a in times['SWAD'][3:])/max(a[unit] for a in times['native'][3:]) for unit in ['wall','CPU']};ratio=max(1.,*ratios.values());reserve=1.25*31189.589142517*ratio+1200
 return dict(times=times,ratio_definition='max measured SWAD arm / max measured native arm, discard first3; conservative max over wall/CPU and1',ratios=ratios,conservative_ratio=ratio,projected_core_based_full6_reserve_seconds=reserve,gate_passed=reserve<=43200,partial_core_profile_not_complete_BN_journal_runner_admission=True)
def main():
 assert not(P/'GENERATED_CORE_RESULT.json').exists();start=time.monotonic();cpu=time.process_time()
 cases=toy_cases();native_result=native_checks();timing=profile()
 result=dict(state='generated_fullnative_SWAD_dense_average_official_corner_state_restore_and_fixed_profile_complete',toy_independent_cases=cases,native=native_result,profile=timing,real_UWB_arrays_labels_or_IDs_opened=False,new_real_fit_started=False,complete_LOSO6_runner_or_BN_calibration_implemented=False,wall=time.monotonic()-start,CPU=time.process_time()-cpu)
 (P/'GENERATED_CORE_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(state=result['state'],native=native_result,resource_gate_passed=timing['gate_passed'],reserve_seconds=timing['projected_core_based_full6_reserve_seconds'],wall=result['wall'],CPU=result['CPU'])))
if __name__=='__main__':main()
