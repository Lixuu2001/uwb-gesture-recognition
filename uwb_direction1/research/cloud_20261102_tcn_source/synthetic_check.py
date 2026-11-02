"""Full TCN graph NumPy inference, causality and complete-state next update."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,tempfile
from native_tcn import np,torch,make,capture,restore,equal,F
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
def conv(x,l):
 if hasattr(l,'weight_v'):
  v=l.weight_v.detach().numpy();g=l.weight_g.detach().numpy();w=v*(g/np.sqrt(np.sum(v*v,axis=(1,2),keepdims=True)))
 else:w=l.weight.detach().numpy()
 bias=l.bias.detach().numpy();out=np.zeros((len(x),len(w),x.shape[2]),dtype='float32')
 for k in range(w.shape[2]):
  lag=l.dilation[0]*(w.shape[2]-1-k)
  if lag<x.shape[2]:out[:,:,lag:]+=np.einsum('nct,oc->not',x[:,:,:x.shape[2]-lag],w[:,:,k],optimize=True)
 return out+bias[None,:,None]
def reference(m,x):
 for b in m.tcn.network:
  out=np.maximum(conv(x,b.conv1),0);out=np.maximum(conv(out,b.conv2),0);res=x if b.downsample is None else conv(x,b.downsample);x=np.maximum(out+res,0)
 z=x[:,:,-1]@m.linear.weight.detach().numpy().T+m.linear.bias.detach().numpy();z=z-z.max(axis=1,keepdims=True)
 return z-np.log(np.exp(z).sum(axis=1,keepdims=True))
def update(m,opt,x,y):
 m.train();opt.zero_grad();loss=F.nll_loss(m(x),y);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters());opt.step();return float(loss.detach())
def main():
 g=np.random.default_rng(54);x=torch.tensor(g.normal(size=(2,567,90)).astype('float32'));y=torch.tensor([0,1]);m,opt=make(20261029);initial=capture(m,opt)
 assert len(m.tcn.network)==8 and [b.conv1.dilation[0] for b in m.tcn.network]==[1,2,4,8,16,32,64,128]
 assert all(b.conv1.out_channels==25 and b.conv2.kernel_size==(7,) and b.dropout1.p==b.dropout2.p==.05 for b in m.tcn.network)
 for _ in range(3):assert np.isfinite(update(m,opt,x,y))
 saved=capture(m,opt);assert len(opt.state)==len(list(m.parameters()))
 m.eval()
 with torch.no_grad():p=m(x).numpy();whole=m.tcn(x);future=x.clone();future[:,:,41:]=100;altered=m.tcn(future)
 np.testing.assert_array_equal(whole[:,:,:41],altered[:,:,:41]);np.testing.assert_allclose(p,reference(m,x.numpy()),rtol=5e-5,atol=5e-6);np.testing.assert_allclose(np.exp(p).sum(axis=1),1,atol=2e-7)
 restore(saved,m,opt)
 with tempfile.TemporaryDirectory() as d:
  path=Path(d)/'full.pt';torch.save(saved,path);fresh,fo=make(20261029);restore(torch.load(path,weights_only=False),fresh,fo);equal(capture(fresh,fo),saved)
  lb=update(fresh,fo,x,y);after=capture(fresh,fo);restore(saved,m,opt);la=update(m,opt,x,y);equal(la,lb);equal(capture(m,opt),after)
 restore(initial,m,opt);equal(capture(m,opt),initial)
 # Source init_weights writes legacy derived .weight, not weight_g/v. The
 # first forward hook recomputes from g/v; preserve and disclose this behavior.
 fresh,fo=make(7);b=fresh.tcn.network[0];pre=b.conv1.weight.detach().clone()
 with torch.no_grad():fresh(x)
 derived_changed=not torch.equal(pre,b.conv1.weight.detach());assert derived_changed
 r=dict(state='pass_full_author_released_TCN_CLI_synthetic_only',total_parameters=sum(p.numel() for p in m.parameters()),trainable_tensors=len(list(m.parameters())),state_dict_keys=len(m.state_dict()),blocks=8,channels=25,kernel=7,dropout=.05,causal_two_convolutions_per_block=True,last_timestep_linear_logsoftmax=True,synthetic_updates=3,independent_NumPy_full_graph_inference=True,future_input_prefix_invariance_exact=True,complete_weights_Adam_modes_global_RNG_fresh_restore_next_update_exact=True,initial_restore_exact=True,legacy_weight_norm_init_derived_cache_recomputed_on_first_forward=True,source_default_epochs=20,source_default_batch_size=64,source_default_training_shuffle=False,source_default_checkpoint_selection='final_epoch20',source_LR_decays_after_epochs=[10,20],UWB_values_labels_fit=False,formal_cost_run=False,original_PyTorch_numeric_parity_claim=False,novelty_claim=False)
 Path(__file__).with_name('SYNTHETIC_CHECK.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
