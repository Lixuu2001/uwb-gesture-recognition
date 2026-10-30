"""Synthetic validation of the literal CLI defaults, not an accuracy experiment."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import io,json,time
import numpy as np
import torch
from author_core import BASE,source_check,optimizer
from cli_core import make_model,cli_defaults
from synthetic_check import equal
def main():
 assert not (BASE/'CLI_SYNTHETIC_CHECK.json').exists();started=time.monotonic();source_check()
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 model,ns,cfg=make_model();opt=optimizer(model,ns);criterion=ns['NoFussCrossEntropyLoss'](reduction='none')
 pos=np.arange(90)[:,None];f=10000.**(-2*np.arange(8)/16)*16/90;pe=np.empty((90,16));pe[:,0::2]=np.sin(pos*f);pe[:,1::2]=np.cos(pos*f)
 np.testing.assert_allclose(model.Fix_Position.pe.numpy()[0],pe,rtol=1e-5,atol=2e-6)
 data=torch.randn(2,567,90,generator=torch.Generator().manual_seed(993));y=torch.tensor([0,1]);initial={n:p.detach().clone() for n,p in model.named_parameters()}
 def update(m,o):
  m.train();o.zero_grad(set_to_none=True);logits=m(data);assert logits.shape==(2,12) and torch.isfinite(logits).all();loss=criterion(logits,y).mean();loss.backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters());torch.nn.utils.clip_grad_norm_(m.parameters(),4.);o.step();return float(loss.detach())
 for _ in range(6): update(model,opt)
 changed=sum(not torch.equal(p,initial[n]) for n,p in model.named_parameters());assert changed==len(initial)
 buffer=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state()),buffer);buffer.seek(0);saved=torch.load(buffer,weights_only=False)
 loss=[update(model,opt) for _ in range(2)];end_rng=torch.get_rng_state();other,ns2,cfg2=make_model();other.load_state_dict(saved['model']);opt2=optimizer(other,ns2);opt2.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng'])
 assert [update(other,opt2) for _ in range(2)]==loss;equal(model.state_dict(),other.state_dict());equal(opt.state_dict(),opt2.state_dict());assert torch.equal(end_rng,torch.get_rng_state())
 result=dict(state='pass',synthetic_only=True,real_data_or_labels_read=False,real_training_started=False,config=cfg,parameters=sum(p.numel() for p in model.parameters()),source_CLI_defaults=cli_defaults(),all_gradients_finite=True,changed_parameter_tensors=changed,full_model_batch=2,optimizer_steps=8,stochastic_resume_steps_exact=2,all_model_optimizer_rng_BN_states_exact=True,tAPE_closed_form_verified=True,wall_seconds=time.monotonic()-started)
 (BASE/'CLI_SYNTHETIC_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__': main()
