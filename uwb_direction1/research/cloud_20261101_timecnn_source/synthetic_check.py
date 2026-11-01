"""Full-size native Time-CNN synthetic and independent NumPy references."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,tempfile
from pathlib import Path
from native_timecnn import np,torch,keras,make,capture,restore,equal
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
def sigmoid(x):return 1/(1+np.exp(-x))
def reference(m,x):
 for layer in m.layers:
  kind=layer.__class__.__name__;w=layer.get_weights()
  if kind=='InputLayer':continue
  if kind=='Conv1D':
   assert layer.padding=='valid';k=w[0].shape[0];z=np.lib.stride_tricks.sliding_window_view(x,k,axis=1).transpose(0,1,3,2);x=sigmoid(np.einsum('ntkc,kco->nto',z,w[0],optimize=True)+w[1])
  elif kind=='AveragePooling1D':
   size=layer.pool_size[0];stride=layer.strides[0];x=np.stack([x[:,i:i+size].mean(axis=1) for i in range(0,x.shape[1]-size+1,stride)],axis=1)
  elif kind=='Flatten':x=x.reshape(len(x),-1)
  elif kind=='Dense':x=sigmoid(x@w[0]+w[1])
  else:raise AssertionError(kind)
 return x
def main():
 g=np.random.default_rng(54);x=g.normal(size=(4,90,567)).astype('float32');y=np.eye(12,dtype='float32')[[0,1,2,3]]
 with tempfile.TemporaryDirectory() as d:
  h,m=make(d,20261029);assert m.count_params()==25356 and len(m.trainable_weights)==6 and len(m.optimizer.variables)==14 and m.loss=='mean_squared_error'
  assert [(z.filters,z.kernel_size,z.padding,z.activation.__name__) for z in m.layers if isinstance(z,keras.layers.Conv1D)]==[(6,(7,),'valid','sigmoid'),(12,(7,),'valid','sigmoid')]
  assert m.layers[-1].activation.__name__=='sigmoid' and len(h.callbacks)==1 and h.callbacks[0].monitor=='loss' and h.callbacks[0].save_best_only
  initial=capture(m)
  for _ in range(3):
   m.reset_metrics();logs=m.train_on_batch(x,y,return_dict=True);assert all(np.isfinite(float(v)) for v in logs.values())
  assert int(m.optimizer.iterations)==3;saved=capture(m)
  with torch.no_grad():p=m(x,training=False).detach().numpy()
  ref=reference(m,x);np.testing.assert_allclose(p,ref,rtol=2e-5,atol=2e-6);assert not np.allclose(p.sum(axis=1),1)
  m.reset_metrics();logs=m.test_on_batch(x,y,return_dict=True);np.testing.assert_allclose(logs['loss'],np.square(ref-y).mean(),rtol=2e-5,atol=2e-6)
  restore(saved,m);torch.save(saved,Path(d)/'full.pt');m.save(Path(d)/'native.keras');loaded=keras.models.load_model(Path(d)/'native.keras');equal(loaded.get_weights(),saved['native']['weights']);equal([v.numpy() for v in loaded.optimizer.variables],saved['native']['optimizer'])
  _,fresh=make(d,20261029);restore(torch.load(Path(d)/'full.pt',weights_only=False),fresh);equal(capture(fresh),saved);fresh.reset_metrics();lb=fresh.train_on_batch(x,y,return_dict=True);next_state=capture(fresh)
  restore(saved,m);m.reset_metrics();la=m.train_on_batch(x,y,return_dict=True);equal(la,lb);equal(capture(m),next_state);restore(initial,m);equal(capture(m),initial)
  r={'state':'pass_full_review_TimeCNN_native_synthetic_only','total_parameters':25356,'trainable_parameters':25356,'trainable_tensors':6,'Adam_variables':14,'input_shape':[90,567],'full_review_filters':[6,12],'kernels':[7,7],'pool_size':3,'all_activations_sigmoid':True,'loss':'mean_squared_error','output_is_not_probability_simplex':True,'source_default_epochs':2000,'source_default_batch_size':16,'synthetic_updates':3,'full_NumPy_Conv_AvgPool_Flatten_sigmoid_and_MSE_reference':True,'full_state_fresh_restore_next_update_exact':True,'native_weights_Adam_reload_exact':True,'initial_restoration_exact':True,'formal_cost_run':False,'UWB_values_labels_fit':False,'original_paper_TF_parity':False,'accuracy_measured':False,'novelty_claim':False};Path(__file__).with_name('SYNTHETIC_CHECK.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
