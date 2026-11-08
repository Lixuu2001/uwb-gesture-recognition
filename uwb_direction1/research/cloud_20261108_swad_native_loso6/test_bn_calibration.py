"""All-six generated roles: input firewall, independent BN stats, exact RNG/mode restoration."""
from pathlib import Path
import copy,sys,json,time,random
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P/'shared'))
import engine as e
from common import read,write
from bn_calibration import calibrate

def main():
 assert not(P/'BN_SYNTHETIC_RESULT.json').exists();started=time.monotonic();cpu=time.process_time();rows=[];timings=[];evals=[];maxmean=0.;maxvar=0.
 for h in range(1,7):
  cfg=read(P/f'folds/held{h}/PROTOCOL.json');n=[69,71,78][(h-1)%3];groups=cfg['train_users']
  users=e.np.concatenate([e.np.full(n,u,dtype='int64') for u in groups]+[e.np.full(64,cfg['validation_user'],dtype='int64'),e.np.full(64,h,dtype='int64')]);ids=e.np.arange(len(users),dtype='int64')
  positions=e.np.flatnonzero(e.np.isin(users,groups));val=e.np.flatnonzero(users==cfg['validation_user']);held=e.np.flatnonzero(users==h)
  values=e.np.random.default_rng(551+h).normal(size=(len(users),90,567)).astype('float32')
  class Maps:
   def __init__(self):self.allowed=positions.copy();self.opened=[]
   def __len__(self):return len(values)
   def __getitem__(self,ix):
    ix=e.np.asarray(ix);assert e.np.isin(ix,self.allowed).all();self.opened.extend(ix.tolist());return values[ix]
  maps=Maps();m,o,g=e.make();m.eval();initial=copy.deepcopy(m.state_dict());expected={};handles=[]
  for name,mod in m.named_modules():
   if isinstance(mod,e.torch.nn.modules.batchnorm._BatchNorm):
    expected[name]=[]
    def hook(layer,args,name=name):
     a=args[0].detach().numpy().astype('float64');axes=(0,)+tuple(range(2,a.ndim));expected[name].append((a.mean(axis=axes),a.var(axis=axes,ddof=1)))
    handles.append(mod.register_forward_pre_hook(hook))
  assert expected;before=e.semantic(m.state_dict())
  # Role guard must reject BEFORE any parameter/buffer/mode or RNG mutation.
  bad=positions.copy();bad[-1]=held[0]
  try:calibrate(m,maps,bad,ids,users,groups)
  except AssertionError:pass
  else:raise AssertionError('Held input admitted to calibration')
  assert before==e.semantic(m.state_dict())
  original_momenta={k:x.momentum for k,x in m.named_modules() if k in expected};modes={k:x.training for k,x in m.named_modules()};rng=e.torch.get_rng_state().clone()
  meta,hook_cost=calibrate(m,maps,positions,ids,users,groups)
  assert e.torch.equal(rng,e.torch.get_rng_state()) and modes=={k:x.training for k,x in m.named_modules()} and original_momenta=={k:x.momentum for k,x in m.named_modules() if k in expected}
  assert sorted(maps.opened)==positions.tolist() and len(maps.opened)==len(set(maps.opened));assert not set(maps.opened)&set(val.tolist()+held.tolist())
  for handle in handles:handle.remove()
  for name,obs in expected.items():
   mod=dict(m.named_modules())[name];mean=e.np.stack([a for a,b in obs]).mean(0);var=e.np.stack([b for a,b in obs]).mean(0)
   dm=float(abs(mod.running_mean.numpy().astype('float64')-mean).max());dv=float(abs(mod.running_var.numpy().astype('float64')-var).max());maxmean=max(maxmean,dm);maxvar=max(maxvar,dv)
   e.np.testing.assert_allclose(mod.running_mean.numpy(),mean,rtol=3e-5,atol=3e-6);e.np.testing.assert_allclose(mod.running_var.numpy(),var,rtol=3e-5,atol=3e-6)
   assert int(mod.num_batches_tracked)==len(obs)==(len(positions)+63)//64
  calibrated=copy.deepcopy(m.state_dict());m.load_state_dict(initial,strict=True);maps.opened=[]
  # Fixed unhooked repeat gives exact common-random-number calibration and
  # production-only cost; no timing is chosen from reference-hook work.
  meta2,cost=calibrate(m,maps,positions,ids,users,groups);assert meta==meta2 and e.semantic(calibrated)==e.semantic(m.state_dict())
  assert sorted(maps.opened)==positions.tolist()
  for a in cost['forward_batches']:
   if a['n']==64:timings.append(a)
  maps.allowed=val;maps.opened=[];before=e.semantic(m.state_dict());old=len(e.EVAL_BATCH_COST);z=e.evaluate(m,maps,val)
  assert before==e.semantic(m.state_dict()) and z.shape==(64,12) and maps.opened==val.tolist();evals.extend(e.EVAL_BATCH_COST[old:])
  rows.append(dict(held=h,train_source_count=len(positions),tail=len(positions)%64,BN_modules=len(expected),source_only_input_firewall=True,all_training_IDs_exactly_once=True,independent_NumPy_batch_mean_and_unbiased_variance=True,repeat_bitwise_exact=True,parameters_and_RNG_modes_momenta_unchanged=True,reference_hook_cost=hook_cost,production_unhooked_cost=cost,calibration_metadata=meta,held_or_validation_inputs_calibrated=False))
  del values,m,o,g,maps
 result=dict(state='pass_all6_generated_fullnative_BN_roles_independent_stats_repeat_RNG_modes',rows=rows,max_mean_absolute_difference=maxmean,max_variance_absolute_difference=maxvar,production_full_B64_BN_batch_timings=timings,production_full_B64_eval_batch_timings=evals,BN_tolerance=dict(rtol=3e-5,atol=3e-6),actual_UWB_arrays_labels_or_IDs_read=False,newfit_started=False,wall=time.monotonic()-started,CPU=time.process_time()-cpu)
 write(P/'BN_SYNTHETIC_RESULT.json',result);print(json.dumps(dict(state=result['state'],max_mean_absolute_difference=maxmean,max_var_absolute_difference=maxvar,wall=result['wall'],CPU=result['CPU'])))
if __name__=='__main__':main()
