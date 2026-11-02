"""Full native model and independent NumPy inference, synthetic inputs only."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,tempfile
from pathlib import Path
from native_fcn import np,torch,keras,make,capture,restore,equal,check_source
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
def reference(m,x):
 for layer in m.layers:
  kind=layer.__class__.__name__;w=layer.get_weights()
  if kind=='InputLayer':continue
  if kind=='Conv1D':
   k=w[0].shape[0];p=(k-1)//2;z=np.pad(x,((0,0),(p,k-1-p),(0,0)))
   z=np.lib.stride_tricks.sliding_window_view(z,k,axis=1).transpose(0,1,3,2)
   x=np.einsum('ntkc,kco->nto',z,w[0],optimize=True)+w[1]
  elif kind=='BatchNormalization':x=(x-w[2])/np.sqrt(w[3]+np.float32(layer.epsilon))*w[0]+w[1]
  elif kind=='Activation':x=np.maximum(x,0)
  elif kind=='GlobalAveragePooling1D':x=x.mean(axis=1)
  elif kind=='Dense':
   x=x@w[0]+w[1];x=np.exp(x-x.max(axis=1,keepdims=True));x=x/x.sum(axis=1,keepdims=True)
  else:raise AssertionError(kind)
 return x
def main():
 check_source();g=np.random.default_rng(53);x=g.normal(size=(4,90,567)).astype('float32');y=np.eye(12,dtype='float32')[[0,1,2,3]]
 with tempfile.TemporaryDirectory() as d:
  h,m=make(d,20261029);assert m.count_params()==846860
  assert sum(int(np.prod(v.shape)) for v in m.trainable_weights)==845836 and len(m.trainable_weights)==14 and len(m.optimizer.variables)==30
  assert [(z.filters,z.kernel_size) for z in m.layers if isinstance(z,keras.layers.Conv1D)]==[(128,(8,)),(256,(5,)),(128,(3,))]
  assert h.callbacks[0].monitor=='loss' and h.callbacks[0].factor==.5 and h.callbacks[0].patience==50 and h.callbacks[0].min_lr==.0001
  assert isinstance(h.callbacks[1],keras.callbacks.ModelCheckpoint) and h.callbacks[1].monitor=='loss' and h.callbacks[1].save_best_only
  opt=keras.optimizers.serialize(m.optimizer)['config'];assert opt['epsilon']==1e-7 and opt['beta_1']==.9 and opt['beta_2']==.999
  initial=capture(m);loss=[]
  for _ in range(3):
   m.reset_metrics();r=m.train_on_batch(x,y,return_dict=True);assert all(np.isfinite(v) for v in r.values());loss.append(r)
  assert int(m.optimizer.iterations)==3
  saved=capture(m);torch.save(saved,Path(d)/'complete_state.pt')
  with torch.no_grad():pred=m(x,training=False).detach().numpy()
  np.testing.assert_allclose(pred,reference(m,x),rtol=2e-4,atol=2e-5)
  # Save/load may consume Python RNG. Restore captured full state before comparison.
  restore(saved,m);m.save(Path(d)/'native.keras');loaded=keras.models.load_model(Path(d)/'native.keras')
  equal(loaded.get_weights(),saved['native']['weights']);equal([v.numpy() for v in loaded.optimizer.variables],saved['native']['optimizer'])
  h2,fresh=make(d,20261029);restore(torch.load(Path(d)/'complete_state.pt',weights_only=False),fresh);equal(capture(fresh),saved)
  fresh.reset_metrics();next_b=fresh.train_on_batch(x,y,return_dict=True);next_state=capture(fresh)
  restore(saved,m);m.reset_metrics();next_a=m.train_on_batch(x,y,return_dict=True);equal(next_a,next_b);equal(capture(m),next_state)
  restore(initial,m);equal(capture(m),initial)
  result={'state':'pass_full_FCN_native_synthetic_preparation','total_parameters':846860,'trainable_parameters':845836,'trainable_tensors':14,'Adam_variables':30,'input_shape':[90,567],'output_classes':12,'complete_source_AST_artifact_extension_only':True,'native_Keras3_Torch_backend':True,'legacy_TF_numeric_parity':False,'synthetic_updates':3,'independent_NumPy_full_Conv_BN_ReLU_GAP_softmax_reference':True,'complete_custom_state_restore_and_next_update_exact':True,'native_weights_Adam_reload_exact':True,'initial_restore_exact':True,'source_callback_order_reduceLR_then_checkpoint_checked':True,'source_default_epochs':2000,'source_default_batch_size':16,'formal_full_budget_cost_passed':False,'UWB_values_labels_fit':False,'accuracy_measured':False,'novelty_claim':False}
  Path(__file__).with_name('SYNTHETIC_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
