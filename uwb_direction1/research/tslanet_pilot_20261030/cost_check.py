"""Single fixed full-author-scale synthetic cost check; no project arrays/labels."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import json,math,resource,signal,sys,time
from pathlib import Path
import torch
sys.path.insert(0,'/workspace/uwb-tslanet-screen-20261029')
from author_core import make_model
from training_ops import pretraining_loss,classification_criterion,stage_optimizer
def main():
 resource.setrlimit(resource.RLIMIT_CPU,(250,260));signal.setitimer(signal.ITIMER_REAL,270);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 started=time.monotonic();times={}
 for phase in ('pretrain','classification'):
  model,_=make_model(20261029);model.train();opt=stage_optimizer(model);criterion=classification_criterion();x=torch.randn(16,567,90);y=torch.arange(16)%12
  def loss():return pretraining_loss(model,x) if phase=='pretrain' else criterion(model(x),y)
  opt.zero_grad();v=loss();v.backward();opt.step() # Fixed one warmup; optimizer allocation covered by reserve.
  training=[]
  for _ in range(3):
   model.train();at=time.monotonic();opt.zero_grad();v=loss();assert torch.isfinite(v);v.backward();opt.step();training.append(time.monotonic()-at)
  validation=[];model.eval()
  with torch.no_grad():
   for _ in range(3):
    at=time.monotonic();v=loss();assert torch.isfinite(v);validation.append(time.monotonic()-at)
  times[phase]=dict(training=training,validation=validation,max_training=max(training),max_validation=max(validation));del model,opt
 batches=4700//16;val=math.ceil(1000/16);diag=math.ceil(1200/16)
 estimate=2*(50*batches*times['pretrain']['max_training']+200*batches*times['classification']['max_training']+50*val*times['pretrain']['max_validation']+200*val*times['classification']['max_validation']+2*diag*times['classification']['max_validation'])+120
 result=dict(state='pass' if estimate<=43200 else 'resource_gate_failed',synthetic_only=True,real_data_or_labels_read=False,parameters=4132879,embedding=256,depth=3,batch_size=16,pretraining_epochs=50,classification_epochs_per_arm=100,timings=times,estimated_seconds=estimate,cpu_budget_seconds=43200,wall_budget_seconds=43200,margin=2,reserve_seconds=120,wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time())
 p=Path(__file__).with_name('COST_CHECK.json');assert not p.exists();p.write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
