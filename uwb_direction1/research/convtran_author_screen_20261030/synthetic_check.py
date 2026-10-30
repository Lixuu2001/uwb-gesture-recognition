"""Synthetic-only full-width gradients/resume and independent position numerics."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import copy,io,json,math,time
from pathlib import Path
import numpy as np
import torch
from author_core import BASE,source_check,definitions,make_model,optimizer
def equal(a,b):
 if isinstance(a,torch.Tensor): assert torch.equal(a,b)
 elif isinstance(a,dict):
  assert set(a)==set(b)
  for k in a: equal(a[k],b[k])
 elif isinstance(a,(tuple,list)):
  assert type(a)==type(b) and len(a)==len(b)
  for x,y in zip(a,b): equal(x,y)
 else: assert a==b
def main():
 assert not (BASE/'SYNTHETIC_CHECK.json').exists()
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 started=time.monotonic();index=source_check();ns=definitions()
 # Independent closed-form tAPE and eRPE computations, with nonzero learned bias.
 pe=ns['tAPE'](64,dropout=0,max_len=90).pe.numpy()[0]
 pos=np.arange(90,dtype=np.float64)[:,None];freq=10000.**(-2*np.arange(32,dtype=np.float64)/64)*64/90
 expected=np.empty((90,64));expected[:,0::2]=np.sin(pos*freq);expected[:,1::2]=np.cos(pos*freq)
 np.testing.assert_allclose(pe,expected,rtol=1e-5,atol=3e-6)
 torch.manual_seed(812);attention=ns['Attention_Rel_Scl'](64,8,9,.01).double().eval()
 with torch.no_grad(): attention.relative_bias_table.copy_(torch.linspace(-.1,.2,17*8).reshape(17,8))
 x=torch.randn(3,9,64,dtype=torch.float64);actual=attention(x).detach().numpy();inp=x.numpy()
 q=(inp@attention.query.weight.detach().numpy().T).reshape(3,9,8,8).transpose(0,2,1,3)
 k=(inp@attention.key.weight.detach().numpy().T).reshape(3,9,8,8).transpose(0,2,3,1)
 v=(inp@attention.value.weight.detach().numpy().T).reshape(3,9,8,8).transpose(0,2,1,3)
 score=q@k*64**-.5;weights=np.exp(score-score.max(-1,keepdims=True));weights/=weights.sum(-1,keepdims=True)
 relative=np.arange(9)[:,None]-np.arange(9)[None,:]+8;np.testing.assert_array_equal(attention.relative_index.numpy().reshape(9,9),relative)
 bias=attention.relative_bias_table.detach().numpy()[relative].transpose(2,0,1)[None]
 output=((weights+bias)@v).transpose(0,2,1,3).reshape(3,9,64)
 expected=(output-output.mean(-1,keepdims=True))/np.sqrt(output.var(-1,keepdims=True)+attention.to_out.eps)
 expected=expected*attention.to_out.weight.detach().numpy()+attention.to_out.bias.detach().numpy()
 np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)
 # Original loss wrapper versus independent CE.
 criterion=ns['NoFussCrossEntropyLoss'](reduction='none');z=torch.randn(5,12);y=torch.arange(5,dtype=torch.float32)
 torch.testing.assert_close(criterion(z,y).mean(),torch.nn.functional.cross_entropy(z,y.long()),rtol=0,atol=0)
 del attention,x
 generator=torch.Generator().manual_seed(771);data=torch.randn(2,567,90,generator=generator);labels=torch.tensor([0,1])
 model,namespace,cfg=make_model();opt=optimizer(model,namespace);loss_fn=namespace['NoFussCrossEntropyLoss'](reduction='none')
 parameter_count=sum(p.numel() for p in model.parameters());initial={n:p.detach().clone() for n,p in model.named_parameters()}
 def update(m,o):
  m.train();o.zero_grad(set_to_none=True);out=m(data);assert out.shape==(2,12) and torch.isfinite(out).all()
  loss=loss_fn(out,labels).mean();assert torch.isfinite(loss);loss.backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
  torch.nn.utils.clip_grad_norm_(m.parameters(),4.);o.step();return float(loss.detach())
 losses=[update(model,opt) for _ in range(6)]
 assert {v['step'] for v in opt.state.values()}=={6}
 changed=sum(not torch.equal(initial[n],p) for n,p in model.named_parameters());assert changed==len(initial);del initial
 blob=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state()),blob);blob.seek(0);saved=torch.load(blob,weights_only=False)
 next_losses=[update(model,opt) for _ in range(2)];end_rng=torch.get_rng_state()
 resumed,other_namespace,_=make_model();resumed.load_state_dict(saved['model']);other=optimizer(resumed,other_namespace);other.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng'])
 replay_losses=[update(resumed,other) for _ in range(2)]
 assert next_losses==replay_losses;equal(model.state_dict(),resumed.state_dict());equal(opt.state_dict(),other.state_dict());assert torch.equal(end_rng,torch.get_rng_state())
 assert all(int(v)==8 for n,v in model.state_dict().items() if n.endswith('num_batches_tracked'))
 result=dict(state='pass',synthetic_only=True,real_UWB_values_or_labels_read=False,real_training_started=False,author_commit=index['commit'],author_source_files_verified=len(index['files']),dependency_files_verified=len(index['dependencies']),config=cfg,parameters=parameter_count,all_parameter_gradients_finite=True,changed_parameter_tensors=changed,optimizer='unmodified_author_RAdam_lr.001_weightdecay0',gradient_clip_norm=4.,full_model_training_batch=2,synthetic_training_steps_before_resume=6,stochastic_resume_steps_exact=2,all_model_optimizer_rng_and_BN_states_exact=True,tAPE_NumPy_closed_form_max_abs_error=float(np.max(np.abs(pe-np.stack([np.sin(pos*freq),np.cos(pos*freq)],-1).reshape(90,64)))),eRPE_nonzero_bias_NumPy_forward_verified=True,embedding_wide_attention_scale_preserved=True,post_softmax_bias_preserved=True,CE_mean_matches_author_wrapper=True,wall_seconds=time.monotonic()-started,limitations='Synthetic full-width author core test, not real UWB training, complete paper reproduction, resource gate or accuracy evidence. Paper/code and CLI-width differences documented separately.')
 (BASE/'SYNTHETIC_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__': main()
