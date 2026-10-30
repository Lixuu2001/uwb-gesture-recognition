"""Independent native component math, full architecture and state restoration; no UWB."""
import copy,json,tempfile,time
from pathlib import Path
from author_adapter import *
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic();rng=np.random.RandomState(21)
 # TensorFlow SAME-even convention and population BN, checked independently in NumPy.
 x=rng.randn(4,17,3).astype('float32');conv=keras.layers.Conv1D(5,8,padding='same');conv(x);w=rng.randn(8,3,5).astype('float32')*.1;bias=rng.randn(5).astype('float32')*.1;conv.set_weights([w,bias]);pad=np.pad(x,((0,0),(3,4),(0,0)));ref=np.stack([np.einsum('bkc,kco->bo',pad[:,i:i+8],w)+bias for i in range(17)],axis=1);np.testing.assert_allclose(conv(x).detach().numpy(),ref,rtol=2e-5,atol=2e-6)
 bn=keras.layers.BatchNormalization();got=bn(x,training=True).detach().numpy();mean=x.mean((0,1));var=x.var((0,1));np.testing.assert_allclose(got,(x-mean)/np.sqrt(var+.001),rtol=2e-5,atol=2e-6);np.testing.assert_allclose(bn.moving_mean.numpy(),.01*mean,rtol=2e-5,atol=2e-6);np.testing.assert_allclose(bn.moving_variance.numpy(),.99+.01*var,rtol=2e-5,atol=2e-6)
 # Independent Adam state/update equation, including Keras epsilon-hat placement.
 p=keras.Variable(np.array([1.,-2.],dtype='float64'));opt=keras.optimizers.Adam();m=np.zeros(2);v=np.zeros(2);expected=np.array([1.,-2.]);lr=float(opt.learning_rate.numpy())
 for t,g in enumerate([np.array([.3,-.4]),np.array([-.2,.7]),np.array([.1,.1])],1):
  m=.9*m+.1*g;v=.999*v+.001*g*g;alpha=lr*np.sqrt(1-.999**t)/(1-.9**t);expected-=alpha*m/(np.sqrt(v)+opt.epsilon);opt.apply_gradients([(torch.from_numpy(g),p)]);np.testing.assert_allclose(p.numpy(),expected,rtol=1e-10,atol=1e-10)
 with tempfile.TemporaryDirectory() as d:
  h,model=make(d);assert keras.__version__=='3.11.3';counts={k:sum(isinstance(l,getattr(keras.layers,k)) for l in model.layers) for k in ['Conv1D','BatchNormalization','Add','GlobalAveragePooling1D','Dense']};assert counts==dict(Conv1D=11,BatchNormalization=12,Add=3,GlobalAveragePooling1D=1,Dense=1)
  assert [l.filters for l in model.layers if isinstance(l,keras.layers.Conv1D)]==[64,64,64,64,128,128,128,128,128,128,128]
  assert model.loss=='categorical_crossentropy' and model.optimizer.get_config()['epsilon']==1e-7 and not model.jit_compile
  x=rng.randn(64,90,567).astype('float32');y=np.eye(12,dtype='float32')[np.arange(64)%12]
  for _ in range(3):result=step(model,x,y);assert np.isfinite(result['loss'])
  saved=copy.deepcopy(snapshot(model));assert int(model.optimizer.iterations.numpy())==3
  for _ in range(2):step(model,x,y)
  final=copy.deepcopy(snapshot(model));h2,other=make(d);restore(saved,other)
  for _ in range(2):step(other,x,y)
  equal(final,snapshot(other));out=other.predict_on_batch(x);assert out.shape==(64,12) and np.isfinite(out).all();np.testing.assert_allclose(out.sum(1),1,rtol=1e-6,atol=1e-6)
  # Native whole-model serialization retains optimizer slots as well as BN state.
  other.save(Path(d)/'native.keras');loaded=keras.models.load_model(Path(d)/'native.keras');equal(other.get_weights(),loaded.get_weights());equal([v.numpy() for v in other.optimizer.variables],[v.numpy() for v in loaded.optimizer.variables])
  result=dict(state='pass',framework='Keras3.11.3nativeTorchbackend',full_author_architecture=True,parameters=model.count_params(),trainable_parameter_tensors=len(model.trainable_variables),native_optimizer_variables=len(model.optimizer.variables),layer_counts=counts,independent_NumPy_SAME_even_conv=True,independent_NumPy_population_BN_and_moving_moments=True,independent_NumPy_Keras_Adam=True,three_native_full_batch_updates=True,next_two_updates_weights_optimizer_BN_metrics_all_RNG_exact=True,native_keras_save_load_weights_optimizer_exact=True,original_TensorFlow_training_reproduction=False,fit_full_epoch_shuffle_parity_tested=False,author_epochs1500_not_run=True,real_UWB_values_labels_or_fit=False,wall_seconds=time.monotonic()-start)
 (B/'SYNTHETIC_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
