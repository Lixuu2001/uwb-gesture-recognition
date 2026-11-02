"""Full Encoder synthetic check, independent NumPy graph and IN derivatives."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,tempfile
from pathlib import Path
from native_encoder import np,torch,keras,make,capture,restore,equal,InstanceNormalization
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
def softmax(x):
 z=np.exp(x-x.max(axis=-1,keepdims=True));return z/z.sum(axis=-1,keepdims=True)
def reference(m,x):
 values={}
 for layer in m.layers:
  kind=layer.__class__.__name__;w=layer.get_weights()
  parents=[values[id(t)] for t in (layer.input if isinstance(layer.input,list) else [layer.input])] if kind!='InputLayer' else []
  if kind=='InputLayer':out=x
  else:
   a=parents[0]
   if kind=='Conv1D':
    k=w[0].shape[0];a=np.pad(a,((0,0),((k-1)//2,k//2),(0,0)))
    z=np.lib.stride_tricks.sliding_window_view(a,k,axis=1).transpose(0,1,3,2)
    out=np.einsum('ntkc,kco->nto',z,w[0],optimize=True)+w[1]
   elif kind=='InstanceNormalization':out=(a-a.mean(axis=1,keepdims=True))/np.sqrt(a.var(axis=1,keepdims=True)+.001)*w[0]+w[1]
   elif kind=='PReLU':out=np.maximum(a,0)+w[0]*np.minimum(a,0)
   elif kind=='Dropout':out=a
   elif kind=='MaxPooling1D':out=np.stack([a[:,i:i+2].max(axis=1) for i in range(0,a.shape[1]-1,2)],axis=1)
   elif kind=='ChannelSlice':out=a[:,:,layer.start:layer.stop]
   elif kind=='Softmax':assert layer.axis==-1;out=softmax(a)
   elif kind=='Multiply':out=parents[0]*parents[1]
   elif kind=='Dense':
    out=a@w[0]+w[1];out=softmax(out) if layer.activation.__name__=='softmax' else 1/(1+np.exp(-out))
   elif kind=='Flatten':out=a.reshape(len(a),-1)
   else:raise AssertionError(kind)
  values[id(layer.output)]=out
 return values[id(m.output)]
def norm_derivatives(g):
 x=g.normal(size=(2,9,5)).astype('float32');up=g.normal(size=x.shape).astype('float32')
 gamma=g.normal(size=5).astype('float32');beta=g.normal(size=5).astype('float32')
 layer=InstanceNormalization();xt=torch.tensor(x,requires_grad=True);out=layer(xt);layer.gamma.assign(gamma);layer.beta.assign(beta);out=layer(xt)
 loss=(out*torch.tensor(up)).sum();loss.backward()
 c=x-x.mean(axis=1,keepdims=True);inv=1/np.sqrt(x.var(axis=1,keepdims=True)+.001);z=c*inv;u=up*gamma
 dx=inv*(u-u.mean(axis=1,keepdims=True)-z*(u*z).mean(axis=1,keepdims=True))
 np.testing.assert_allclose(out.detach().numpy(),z*gamma+beta,rtol=2e-5,atol=2e-6)
 for actual,expected in [(xt.grad.numpy(),dx),(layer.gamma.value.grad.numpy(),(up*z).sum(axis=(0,1))),(layer.beta.value.grad.numpy(),up.sum(axis=(0,1)))]:np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=3e-6)
def main():
 g=np.random.default_rng(54);norm_derivatives(g);x=g.normal(size=(2,90,567)).astype('float32');y=np.eye(12,dtype='float32')[[0,1]]
 with tempfile.TemporaryDirectory() as d:
  h,m=make(d,20261029);params=m.count_params();tensors=len(m.trainable_weights);adam=len(m.optimizer.variables)
  assert [(z.filters,z.kernel_size,z.padding) for z in m.layers if isinstance(z,keras.layers.Conv1D)]==[(128,(5,),'same'),(256,(11,),'same'),(512,(21,),'same')]
  assert m.input_shape==(None,90,567) and m.output_shape==(None,12)
  assert len([l for l in m.layers if isinstance(l,keras.layers.Dropout)])==3 and len([l for l in m.layers if isinstance(l,InstanceNormalization)])==4
  assert len(h.callbacks)==1 and h.callbacks[0].monitor=='loss' and h.callbacks[0].save_best_only
  assert abs(float(m.optimizer.learning_rate)-1e-5)<1e-11 and m.loss=='categorical_crossentropy'
  initial=capture(m)
  for _ in range(3):
   m.reset_metrics();logs=m.train_on_batch(x,y,return_dict=True);assert all(np.isfinite(float(v)) for v in logs.values())
  assert int(m.optimizer.iterations)==3;saved=capture(m)
  assert len(saved['layer_rng'])==3
  # Keras3.11.3 Torch dropout uses the global Torch RNG on CPU; layer seed
  # variables are still captured, but must not be presumed to increment.
  assert not torch.equal(initial['native']['torch_rng'],saved['native']['torch_rng'])
  with torch.no_grad():p=m(x,training=False).detach().numpy()
  ref=reference(m,x);np.testing.assert_allclose(p,ref,rtol=4e-4,atol=8e-6);np.testing.assert_allclose(p.sum(axis=1),1,atol=2e-7)
  m.reset_metrics();logs=m.test_on_batch(x,y,return_dict=True);np.testing.assert_allclose(logs['loss'],-np.sum(y*np.log(np.clip(ref,1e-7,1-1e-7)),axis=1).mean(),rtol=4e-4,atol=8e-6)
  restore(saved,m);torch.save(saved,Path(d)/'full.pt');m.save(Path(d)/'native.keras');loaded=keras.models.load_model(Path(d)/'native.keras')
  equal(loaded.get_weights(),saved['native']['weights']);equal([v.numpy() for v in loaded.optimizer.variables],saved['native']['optimizer'])
  _,fresh=make(d,20261029);restore(torch.load(Path(d)/'full.pt',weights_only=False),fresh);equal(capture(fresh),saved)
  fresh.reset_metrics();lb=fresh.train_on_batch(x,y,return_dict=True);next_state=capture(fresh)
  restore(saved,m);m.reset_metrics();la=m.train_on_batch(x,y,return_dict=True);equal(la,lb);equal(capture(m),next_state);restore(initial,m);equal(capture(m),initial)
  r=dict(state='pass_full_review_Encoder_native_synthetic_only',total_parameters=params,trainable_parameters=sum(int(np.prod(v.shape)) for v in m.trainable_weights),trainable_tensors=tensors,Adam_variables=adam,input_shape=[90,567],full_review_filters=[128,256,512],kernels=[5,11,21],source_epochs=100,source_batch_size=12,optimizer='Adam1e-5_native_eps1e-7',synthetic_updates=3,full_NumPy_graph_reference=True,independent_IN_output_and_analytical_gradients=True,review_attention_axis=-1,review_attention_elementwise_no_time_sum=True,full_state_fresh_restore_next_update_exact=True,native_weights_Adam_reload_exact=True,layer_rng_preserved_in_custom_state=True,initial_restore_exact=True,formal_cost_run=False,UWB_values_labels_fit=False,original_paper_architecture_or_TF_numeric_parity=False,accuracy_measured=False,novelty_claim=False)
  Path(__file__).with_name('SYNTHETIC_CHECK.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
