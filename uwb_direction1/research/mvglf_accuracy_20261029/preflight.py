import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='3'
import json,time,resource,signal,hashlib,platform
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import MVGLF,preprocess
B=Path(__file__).resolve().parent
resource.setrlimit(resource.RLIMIT_CPU,(870,890))
torch.set_num_threads(3);torch.set_num_interop_threads(1);torch.manual_seed(20261029)
results={};g=torch.Generator().manual_seed(4);x=preprocess(torch.randn(32,3,90,189,generator=g));y=torch.arange(32)%12
for arm in ['full','local']:
 torch.manual_seed(20261029);m=MVGLF(arm);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0005)
 times=[];losses=[]
 for j in range(5):
  t=time.monotonic();opt.zero_grad(set_to_none=True);p=m(x);loss=F.cross_entropy(p,y);assert torch.isfinite(loss);loss.backward()
  for name,param in m.named_parameters():assert param.grad is not None and torch.isfinite(param.grad).all(),name
  opt.step();times.append(time.monotonic()-t);losses.append(float(loss.detach()))
 m.eval()
 with torch.no_grad():
  out=m(x[:4]);assert out.shape==(4,12) and torch.isfinite(out).all();assert torch.equal(preprocess(torch.zeros(2,3,90,189)),torch.zeros(2,3,64,128));torch.testing.assert_close(preprocess(torch.randn(2,3,90,189,generator=g))[:1].square().mean(),torch.tensor(1.),rtol=1e-4,atol=1e-4)
 results[arm]=dict(parameters=sum(p.numel() for p in m.parameters()),training_seconds_batch32=times,median_warm_seconds=float(np.median(times[2:])),finite_all_gradients=True,synthetic_losses=losses)
# Pilot: held1,val2,train3/4/5/6 counts4700; uniform shuffle,147 optimizer batches per epoch.
steps=(4700+31)//32
estimate=1.5*100*steps*sum(z['median_warm_seconds'] for z in results.values())
report=dict(state='pass',synthetic_only=True,real_values_or_labels_read=False,arms=results,epochs=100,train_steps_per_epoch=steps,estimated_two_arm_wall_seconds=estimate,budget_wall_seconds=43200,passed=estimate<43200,torch=torch.__version__,python=platform.python_version(),threads=3,model_sha256=hashlib.sha256((B/'model.py').read_bytes()).hexdigest(),preflight_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(B/'PREFLIGHT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
