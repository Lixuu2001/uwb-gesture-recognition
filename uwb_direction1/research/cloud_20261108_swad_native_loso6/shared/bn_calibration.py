"""One deterministic training-subject-only BN pass. No label input or model choice."""
import hashlib,math,random,time
import numpy as np
import torch

def tensor_hash(items):
 h=hashlib.sha256()
 for k,v in items:
  h.update(str(k).encode());h.update(str((tuple(v.shape),str(v.dtype))).encode());h.update(v.detach().contiguous().cpu().numpy().tobytes())
 return h.hexdigest()
def calibrate(model,maps,positions,ids,users,train_users,seed=21161030,batch=64):
 positions=np.asarray(positions);ids=np.asarray(ids);users=np.asarray(users);groups=tuple(map(int,train_users))
 assert len(groups)==4 and len(set(groups))==4 and set(groups)<=set(range(1,7))
 assert ids.ndim==users.ndim==positions.ndim==1 and len(ids)==len(users)==len(maps)
 assert np.issubdtype(positions.dtype,np.integer) and np.issubdtype(ids.dtype,np.integer) and len(set(ids.tolist()))==len(ids)
 assert set(np.unique(users))==set(range(1,7)) and batch==64 and seed==21161030
 canonical=np.flatnonzero(np.isin(users,groups));np.testing.assert_array_equal(positions,canonical)
 assert len(positions)>0 and set(users[positions])==set(groups)
 assert all(p.device.type=='cpu' for p in model.parameters())
 modules=dict(model.named_modules());bn={k:m for k,m in modules.items() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)};assert bn
 assert all(m.track_running_stats for m in bn.values())
 modes={k:m.training for k,m in modules.items()};momenta={k:m.momentum for k,m in bn.items()}
 params=tensor_hash(model.named_parameters());buffers={k:v.detach().clone() for k,v in model.named_buffers()}
 before_torch=torch.get_rng_state().clone();before_py=random.getstate();before_np=np.random.get_state();wall=time.monotonic();cpu=time.process_time();batch_cost=[]
 order=torch.randperm(len(positions),generator=torch.Generator().manual_seed(seed)).numpy();ordered=positions[order]
 try:
  with torch.random.fork_rng(devices=[]),torch.no_grad():
   torch.manual_seed(seed);model.train()
   for m in bn.values():m.reset_running_stats();m.momentum=None
   for offset in range(0,len(ordered),batch):
    ix=ordered[offset:offset+batch];x=np.array(maps[ix],copy=True);assert x.shape==(len(ix),90,567) and x.dtype==np.float32 and np.isfinite(x).all()
    wt=time.monotonic();ct=time.process_time();z=model(torch.from_numpy(x));assert z.shape==(len(ix),12) and torch.isfinite(z).all()
    batch_cost.append(dict(n=len(ix),wall=time.monotonic()-wt,CPU=time.process_time()-ct))
 finally:
  for k,m in bn.items():m.momentum=momenta[k]
  for k,m in modules.items():m.training=modes[k]
 assert torch.equal(before_torch,torch.get_rng_state()) and before_py==random.getstate()
 after_np=np.random.get_state();assert before_np[0]==after_np[0] and np.array_equal(before_np[1],after_np[1]) and before_np[2:]==after_np[2:]
 assert tensor_hash(model.named_parameters())==params
 permitted={k+'.'+s for k in bn for s in ['running_mean','running_var','num_batches_tracked']}
 changed={k for k,v in model.named_buffers() if not torch.equal(v,buffers[k])};assert changed<=permitted
 counts={k:int(m.num_batches_tracked) for k,m in bn.items()};assert all(n==math.ceil(len(positions)/batch) for n in counts.values())
 meta=dict(state='source_only_BN_fullpass_complete_no_labels_no_parameter_update',seed=seed,batch=batch,passes=1,n=len(positions),source_users=list(groups),source_ids_sha256=hashlib.sha256(ids[positions].tobytes()).hexdigest(),ordered_source_ids_sha256=hashlib.sha256(ids[ordered].tobytes()).hexdigest(),forward_batches=len(batch_cost),BN_batches=counts,BN_buffer_sha256=tensor_hash((k,v) for k,v in model.named_buffers() if k in permitted),parameters_sha256=params,parameters_unchanged=True,all_main_RNG_and_modes_and_momenta_restored=True,only_BN_stat_buffers_changed=True,validation_or_held_inputs_used=False)
 return meta,dict(wall=time.monotonic()-wall,CPU=time.process_time()-cpu,forward_batches=batch_cost)
