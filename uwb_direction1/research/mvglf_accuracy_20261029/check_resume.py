import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='3'
import torch,json,tempfile
from pathlib import Path
from model import MVGLF
B=Path(__file__).resolve().parent;torch.set_num_threads(3);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(20261029)
x=torch.randn(2,3,64,128).contiguous(memory_format=torch.channels_last);y=torch.tensor([0,1])
m=MVGLF('full').to(memory_format=torch.channels_last).train();op=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0005)
def step(m,op):
 op.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(m(x),y);loss.backward();op.step()
step(m,op)
with tempfile.TemporaryDirectory() as d:
 p=Path(d)/'resume.pt';torch.save(dict(model=m.state_dict(),optimizer=op.state_dict(),rng=torch.get_rng_state()),p)
 step(m,op);expected={k:v.clone() for k,v in m.state_dict().items()}
 m2=MVGLF('full').to(memory_format=torch.channels_last).train();op2=torch.optim.AdamW(m2.parameters(),lr=.001,weight_decay=.0005);z=torch.load(p,weights_only=False);m2.load_state_dict(z['model']);op2.load_state_dict(z['optimizer']);torch.set_rng_state(z['rng']);step(m2,op2)
 assert all(torch.equal(v,m2.state_dict()[k]) for k,v in expected.items())
 saved=dict(optimizer_and_rng_resume_bitwise_equal=True,batchnorm_state_restored=True,all_model_tensors_equal=True,synthetic_only=True,real_values_or_labels_read=False)
 (B/'RESUME_CHECK.json').write_text(json.dumps(saved,indent=2));print(json.dumps(saved))
