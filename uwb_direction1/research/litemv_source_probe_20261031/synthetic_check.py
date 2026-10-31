"""Fixed-filter integrity and independent separable/dilated conv mathematics."""
import json,tempfile,time,copy
from pathlib import Path
from native_litemv import *
def step(m,x,y):m.reset_metrics();r=m.train_on_batch(x,y,return_dict=True);assert np.isfinite(r['loss']);return r
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic();rng=np.random.RandomState(9)
 # Independent NumPy separable convolution with dilation, plus even SAME padding.
 x=rng.randn(2,17,2).astype('float32');conv=keras.layers.SeparableConv1D(3,4,padding='same',dilation_rate=2,use_bias=False);conv(x);dw=rng.randn(4,2,1).astype('float32')*.1;pw=rng.randn(1,2,3).astype('float32')*.1;conv.set_weights([dw,pw]);pad=np.pad(x,((0,0),(3,3),(0,0)));depth=np.stack([np.einsum('bkc,kc->bc',pad[:,i+np.arange(4)*2],dw[:,:,0]) for i in range(17)],axis=1);expected=np.einsum('btc,co->bto',depth,pw[0]);np.testing.assert_allclose(conv(x).detach().numpy(),expected,rtol=2e-5,atol=2e-6)
 even=keras.layers.DepthwiseConv1D(2,padding='same',use_bias=False,depthwise_initializer=keras.initializers.Constant(np.tile(np.array([-1.,1.],dtype='float32')[:,None,None],(1,2,1))),trainable=False);pad=np.pad(x,((0,0),(0,1),(0,0)));np.testing.assert_allclose(even(x).detach().numpy(),pad[:,1:]-pad[:,:-1],rtol=2e-5,atol=2e-6)
 with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
  h,m=make(a,7);custom=[l for l in m.layers if isinstance(l,keras.layers.DepthwiseConv1D)];assert len(custom)==17 and all(not l.trainable and not l.trainable_variables and not l.kernel.value.requires_grad for l in custom);fixed=[l.kernel.numpy().copy() for l in custom]
  for l in custom:
   w=l.kernel.numpy();k=w.shape[0]
   if 'increasse' in l.name:pattern=np.where(np.arange(k)%2==0,-1,1)
   elif 'decrease' in l.name:pattern=np.where(np.arange(k)%2==0,1,-1)
   else:
    original=k*2//3;q=original//4;ramp=(np.arange(1,q+1)/q)**2;pattern=np.concatenate([-ramp,-ramp[::-1],2*ramp,2*ramp[::-1],-ramp,-ramp[::-1]])
   np.testing.assert_array_equal(w,np.tile(pattern.astype('float32')[:,None,None],(1,567,1)))
  separable=[l for l in m.layers if isinstance(l,keras.layers.SeparableConv1D)];assert [l.kernel_size[0] for l in separable]==[40,20,10,20,10] and [l.dilation_rate[0] for l in separable]==[1,1,1,2,4];assert h.batch_size==64 and h.n_epochs==2 and h.n_filters==32 and h.keep_track==17
  x=rng.randn(2,90,567).astype('float32');y=np.eye(12,dtype='float32')[[0,11]]
  for _ in range(3):step(m,x,y)
  for l,w in zip(custom,fixed):np.testing.assert_array_equal(l.kernel.numpy(),w)
  saved=capture(m);assert int(m.optimizer.iterations.numpy())==3
  for _ in range(2):step(m,x,y)
  final=capture(m);h2,other=make(b,7);restore(saved,other);equal(saved,capture(other))
  for _ in range(2):step(other,x,y)
  equal(final,capture(other));probs=other.predict_on_batch(x);assert probs.shape==(2,12) and np.isfinite(probs).all();np.testing.assert_allclose(probs.sum(1),1,rtol=1e-5,atol=1e-6)
  other.save(Path(b)/'native.keras');loaded=keras.models.load_model(Path(b)/'native.keras');equal(other.get_weights(),loaded.get_weights());equal([v.numpy() for v in other.optimizer.variables],[v.numpy() for v in loaded.optimizer.variables])
  result=dict(state='pass_native_source_probe_only',source_commit='e62f64393104be8eda415b4d974987028ca68754',model_source_blob='ef4aafa16864dfc7f95c7b312459e68573a86961',architecture_methods_reused=5,tf_keras_attributes_explicitly_rebound=NATIVE_BINDINGS,TensorFlow_stub_used=False,original_TensorFlow_execution_parity_asserted=False,backend='Keras3.11.3TorchCPU',paper_epochs1500_not_run=True,current_constructor_epochs=2,ensemble_LITEMVTime_not_implemented=True,custom_filter_layers=17,all_fixed_kernels_exact_analytic_values=True,custom_kernels_nontrainable_and_unchanged_after3updates=True,separable_kernels=[40,20,10,20,10],separable_dilations=[1,1,1,2,4],total_parameters=m.count_params(),trainable_parameters=sum(int(np.prod(v.shape)) for v in m.trainable_variables),trainable_variable_tensors=len(m.trainable_variables),native_Adam_variables=len(m.optimizer.variables),native3updates_and_restore_next2_model_Adam_BN_metrics_all_RNG_exact=True,native_keras_save_load_weights_optimizer_exact=True,independent_NumPy_dilated_separable_conv_and_even_SAME=True,callbacks_epoch_selection_replay_tested=False,formal_CPU_cost_gate_run=False,real_UWB_configuration_selected=False,real_UWB_values_labels_or_fit=False,synthetic_batch_size=2,synthetic_seed=7,wall_seconds=time.monotonic()-start)
 (B/'SOURCE_PROBE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
