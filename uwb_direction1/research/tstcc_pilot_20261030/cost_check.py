"""Prospective synthetic full-default four-phase CPU/resource gate."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,copy,time,resource,warnings
import numpy as np
import torch
from ops import *
B=Path(__file__).resolve().parent
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);warnings.filterwarnings('ignore',message='Implicit dimension choice')
 begin=time.monotonic();r=np.random.RandomState(9);x=r.randn(256,567,90).astype(np.float32);aug_start=time.monotonic();weak,strong,_=augmented(x);aug_time=time.monotonic()-aug_start;x=torch.from_numpy(x[:128]);a=torch.from_numpy(weak[:128]);b=torch.from_numpy(strong[:128]);y=torch.arange(128)%12
 pretrained=copy.deepcopy(make()[1].state_dict());times={};validation=[]
 for phase in ('ssl','linear','finetune','scratch'):
  cfg,m,t,o,to,s=make(phase=phase,pretrained=pretrained if phase in ('linear','finetune') else None);values=[]
  for i in range(8):
   start=time.monotonic();loss,_=step(m,t,o,to,cfg,x,y,a,b,phase);assert torch.isfinite(loss);values.append(time.monotonic()-start)
  times[phase]=dict(max_seconds=max(values),measured_steps=8)
  if phase!='ssl':
   m.eval()
   with torch.inference_mode():
    for i in range(4):
     start=time.monotonic();m(x);validation.append(time.monotonic()-start)
 train=40*36*sum(v['max_seconds'] for v in times.values());eval_cost=(3*40*8+3*18)*max(validation);augmentation=aug_time*4700/256
 estimate=2*(train+eval_cost+augmentation)+120
 result=dict(state='pass' if estimate<=43200 else 'resource_gate_fail',phases=times,full_default_epochs_each=40,full_default_batch=128,validation_max_batch_seconds=max(validation),full_four_phase_updates=5760,full_four_phase_training_appearances=737280,augmentation256_seconds=aug_time,estimate_seconds=estimate,estimate_formula='2*(40*36*sum(four phase maxstep)+((3*40*8)+(3*18))*maxeval+4700/256*aug256)+120',budget_seconds=43200,wall_seconds=time.monotonic()-begin,max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,real_UWB_values_read=False,real_training_started=False)
 (B/'COST_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
