"""Bounded full official gesture-preset synthetic CPU cost, no real arrays."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,time,resource
import numpy as np
import torch
from author_adapter import make,step
B=Path(__file__).resolve().parent
if __name__=='__main__':
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic();cfg,m,o=make();g=torch.Generator().manual_seed(7);x=torch.randn(16,90,567,generator=g);y=torch.arange(16)%12;training=[];validation=[]
 for i in range(8):
  t=time.monotonic();loss=step(m,o,x,y);assert torch.isfinite(loss);training.append(time.monotonic()-t)
 m.eval()
 with torch.inference_mode():
  for i in range(4):
   t=time.monotonic();z=m(x,torch.ones(16,90),None,None);assert torch.isfinite(z).all();validation.append(time.monotonic()-t)
 full=30*294*max(training);val=(30*63+63+75)*max(validation);estimate=2*(full+val)+120
 result=dict(state='pass' if estimate<=43200 else 'resource_gate_fail',preset='originalUWaveGestureLibrary',model_parameters=sum(p.numel() for p in m.parameters()),batch_size=16,epochs_cap=30,training_batches_per_epoch=294,all4700records_used_each_epoch=True,maximum_full_training_updates=8820,training_max_step_seconds=max(training),validation_max_batch_seconds=max(validation),training_steps_measured=8,eval_batches_measured=4,estimated_seconds=estimate,estimate_formula='2*(30*294*maxtrainstep+(30*63+63+75)*maxevalbatch)+120',budget_seconds=43200,wall_seconds=time.monotonic()-start,max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,real_UWB_values_read=False,real_training_started=False)
 (B/'COST_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
