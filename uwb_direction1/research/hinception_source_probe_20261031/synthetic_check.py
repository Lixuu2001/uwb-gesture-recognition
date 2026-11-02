"""Full H-Inception synthetic state/filter integrity; no cost or UWB data."""
import json,tempfile,time
from native_hinception import *
def step(m,x,y):m.reset_metrics();r=m.train_on_batch(x,y,return_dict=True);assert np.isfinite(r['loss']);return r
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic();g=np.random.RandomState(43)
 with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
  h,m,changes=make(a,7,567);fixed=[l for l in m.layers if isinstance(l,keras.layers.Conv1D) and not l.trainable];assert len(fixed)==17 and h.depth==6 and h.n_filters==32 and h.use_residual and h.use_bottleneck and h.max_cf_length==6 and h.epochs==1500 and h.batch_size==64 and h.kernel_sizes==[40,20,10];assert len([l for l in m.layers if isinstance(l,keras.layers.Add)])==2
  before=[l.kernel.numpy().copy() for l in fixed]
  for layer in fixed:
   w=layer.kernel.numpy();k=w.shape[0];assert w.shape[1:]==(567,1) and not layer.trainable_variables and not layer.kernel.value.requires_grad
   if 'increasse' in layer.name:pattern=np.where(np.arange(k)%2==0,-1,1)
   elif 'decrease' in layer.name:pattern=np.where(np.arange(k)%2==0,1,-1)
   else:
    base=k*2//3;q=base//4;ramp=(np.arange(1,q+1)/q)**2;pattern=np.concatenate([-ramp,-ramp[::-1],2*ramp,2*ramp[::-1],-ramp,-ramp[::-1]])
   np.testing.assert_array_equal(w,np.tile(pattern.astype('float32')[:,None,None],(1,567,1)))
  # Independently check the sum across variables in the author fixed Conv1D,
  # unlike LITEMV's channel-preserving depthwise concatenation.
  small=g.randn(2,17,3).astype('float32');layer=keras.layers.Conv1D(1,2,padding='same',use_bias=False,trainable=False,kernel_initializer=keras.initializers.Constant(np.tile(np.array([-1.,1.],dtype='float32')[:,None,None],(1,3,1))));pad=np.pad(small,((0,0),(0,1),(0,0)));np.testing.assert_allclose(layer(small).detach().numpy(),(pad[:,1:]-pad[:,:-1]).sum(2,keepdims=True),rtol=2e-5,atol=2e-6)
  x=g.randn(2,90,567).astype('float32');y=np.eye(12,dtype='float32')[[0,11]]
  for _ in range(3):step(m,x,y)
  for l,w in zip(fixed,before):np.testing.assert_array_equal(l.kernel.numpy(),w)
  saved=capture(m)
  for _ in range(2):step(m,x,y)
  final=capture(m);h2,other,_=make(b,7,567);restore(saved,other);equal(saved,capture(other))
  for _ in range(2):step(other,x,y)
  equal(final,capture(other));other.save(Path(b)/'native.keras');loaded=keras.models.load_model(Path(b)/'native.keras');equal(other.get_weights(),loaded.get_weights());equal([v.numpy() for v in other.optimizer.variables],[v.numpy() for v in loaded.optimizer.variables]);probs=other.predict_on_batch(x);assert probs.shape==(2,12) and np.isfinite(probs).all()
  result=dict(state='pass_full_author_width_depth_with_explicit_multivariate_input_adaptation_synthetic_only',source_commit='cf4d0e55ee11b7393ae3d317cdf5f22dcc51ba3e',model_blob='bb8c08bf24cd3b3dcdfd90f079403900ab4df1eb',adaptations=changes,input_channels_from1_to567=True,architecture_methods_reused=5,width32_depth6_bottleneck32_residual2_full=True,kernels=[40,20,10],custom_filters=17,custom_Conv1D_sums_channels=True,LITEMV_depthwise_concatenation_used=False,all_fixed_kernels_analytic_exact_nontrainable_unchanged_after3=True,independent_NumPy_sum_channel_fixed_conv_even_SAME=True,total_parameters=m.count_params(),trainable_parameters=sum(int(np.prod(v.shape)) for v in m.trainable_variables),trainable_variable_tensors=len(m.trainable_variables),native_Adam_variables=len(m.optimizer.variables),restore_next2_updates_model_Adam_BN_metrics_all_RNG_exact=True,native_keras_reload_weights_optimizer_exact=True,source_author_epochs=1500,source_batch_size=64,ensemble_HInceptionTime_not_implemented=True,original_TensorFlow_reproduction=False,native_TensorFlow_stub_used=False,synthetic_batch=2,synthetic_seed=7,real_UWB_configuration_selected=False,formal_CPU_cost_gate_run=False,real_UWB_values_labels_or_fit=False,wall_seconds=time.monotonic()-start)
 (B/'SOURCE_PROBE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
