import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='3'
import torch,time,json,hashlib,resource
from pathlib import Path
from model import MVGLF,preprocess
B=Path(__file__).resolve().parent;torch.set_num_threads(3);torch.set_num_interop_threads(1);resource.setrlimit(resource.RLIMIT_CPU,(870,890));torch.manual_seed(4);x=preprocess(torch.randn(32,3,90,189));y=torch.arange(32)%12
r={}
for arm in ['full','local']:
 torch.manual_seed(20261029);ref=MVGLF(arm).eval();m=MVGLF(arm);m.load_state_dict(ref.state_dict());m=m.to(memory_format=torch.channels_last).eval();xx=x.contiguous(memory_format=torch.channels_last)
 with torch.no_grad():torch.testing.assert_close(m(xx[:2]),ref(x[:2]),rtol=1e-4,atol=1e-5)
 m.train();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0005);t=[]
 for j in range(5):
  st=time.monotonic();opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(m(xx),y);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters());opt.step();t.append(time.monotonic()-st)
 r[arm]=dict(times=t,median_seconds=sorted(t[2:])[1],output_parity=True,finite_gradients=True)
total=1.5*100*147*sum(d['median_seconds'] for d in r.values());out=dict(synthetic_only=True,arms=r,estimated100epochs_wall_seconds=total,budget_wall_seconds=43200,full_100_epochs_passed=total<43200,memory_format='channels_last',threads=3,model_sha256=hashlib.sha256((B/'model.py').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(B/'PREFLIGHT_CHANNELS_LAST.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
