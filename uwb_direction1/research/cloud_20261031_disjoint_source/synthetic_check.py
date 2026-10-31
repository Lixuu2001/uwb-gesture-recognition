"""Full default DCNN_2L, independent spatial convolution and weighted CE."""
import json,tempfile,time
from native_disjoint import *
def step(m,x,y,w):m.reset_metrics();r=m.train_on_batch(x,y,sample_weight=w,return_dict=True);assert np.isfinite(r['loss']);return r
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic();g=np.random.RandomState(71)
 with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
  h,m=make(a,7);convs=[l for l in m.layers if isinstance(l,keras.layers.Conv2D)];assert len(convs)==4 and all(l.filters==64 and l.use_bias for l in convs) and [l.kernel_size for l in convs]==[(8,1),(1,567),(5,1),(1,64)];assert len([l for l in m.layers if isinstance(l,keras.layers.BatchNormalization)])==4 and len([l for l in m.layers if isinstance(l,keras.layers.ELU)])==4 and len([l for l in m.layers if isinstance(l,keras.layers.Permute)])==1
  pool=next(l for l in m.layers if isinstance(l,keras.layers.MaxPooling2D));assert pool.pool_size==pool.strides==(3,1) and m.output_shape==(None,12)
  # Independent spatial full-width Conv2D contraction, including biases.
  xs=g.randn(2,5,3,2).astype('float32');conv=keras.layers.Conv2D(4,(1,3),padding='valid');conv(xs);kernel=g.randn(1,3,2,4).astype('float32')*.1;bias=g.randn(4).astype('float32')*.1;conv.set_weights([kernel,bias]);reference=np.einsum('btvc,vco->bto',xs,kernel[0])[:, :,None,:]+bias;np.testing.assert_allclose(conv(xs).detach().numpy(),reference,rtol=2e-5,atol=2e-6)
  labels=np.arange(25)%12;onehot=np.eye(12,dtype='float32')[labels];weights=create_class_weight(onehot);expected={c:max(1.,math.log(2*len(labels)/int((labels==c).sum()))) for c in range(12)};np.testing.assert_allclose([weights[c] for c in range(12)],[expected[c] for c in range(12)],rtol=1e-7,atol=1e-7)
  x=g.randn(2,90,567,1).astype('float32');y=np.eye(12,dtype='float32')[[0,11]];w=np.array([weights[0],weights[11]],dtype='float32')
  initial=capture(m);probabilities=m(x,training=True).detach().numpy();reference_loss=float(np.mean(-np.sum(y*np.log(np.clip(probabilities/probabilities.sum(1,keepdims=True),1e-7,1-1e-7)),axis=1)*w));restore(initial,m);equal(initial,capture(m));loss=step(m,x,y,w)['loss'];assert abs(loss-reference_loss)<2e-5,(loss,reference_loss)
  for _ in range(2):step(m,x,y,w)
  saved=capture(m)
  for _ in range(2):step(m,x,y,w)
  final=capture(m);h2,other=make(b,7);restore(saved,other);equal(saved,capture(other))
  for _ in range(2):step(other,x,y,w)
  equal(final,capture(other));other.save(Path(b)/'native.keras');loaded=keras.models.load_model(Path(b)/'native.keras');equal(other.get_weights(),loaded.get_weights());equal([v.numpy() for v in other.optimizer.variables],[v.numpy() for v in loaded.optimizer.variables]);p=other.predict_on_batch(x);assert p.shape==(2,12) and np.isfinite(p).all();np.testing.assert_allclose(p.sum(1),1,rtol=1e-5,atol=1e-6)
  result=dict(state='pass_full_author_default_DCNN2L_synthetic_only',source_commit='38180ccc1f050a3fee6961f5b4ea3bc132016ab3',model_blob='7d46ed906334b2603e39d123b494e4797f11ec64',architecture_methods_reused=3,native_unqualified_layer_bindings=7,artifact_suffix_changes=ARTIFACT_SUFFIX_CHANGES,input_shape=[90,567,1],source_input_shape_functional_argument_unchanged=True,full_author_filters64_4Conv2D_4BN_4ELU_Permute1_pool3=True,total_parameters=m.count_params(),trainable_parameters=sum(int(np.prod(v.shape)) for v in m.trainable_variables),trainable_variable_tensors=len(m.trainable_variables),native_Adam_variables=len(m.optimizer.variables),source_classweight_mu2_log_formula_independent=True,independent_NumPy_spatial_full_width_conv_with_bias=True,independent_NumPy_sample_weighted_CE=True,restore_after3_next2_model_Adam_BN_metrics_all_RNG_exact=True,native_keras_save_load_weights_optimizer_exact=True,original_TensorFlow_execution_parity_asserted=False,TensorFlow_stub_used=False,source_Main_default_epochs500_not_run=True,source_Main_default_batch8_not_real_config_selected=True,synthetic_batch_size=2,synthetic_seed=7,formal_CPU_cost_gate_run=False,real_UWB_configuration_selected=False,real_UWB_values_labels_or_fit=False,wall_seconds=time.monotonic()-start)
 (B/'SOURCE_PROBE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
