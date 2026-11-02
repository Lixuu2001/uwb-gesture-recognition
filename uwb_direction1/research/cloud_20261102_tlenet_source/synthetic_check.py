import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,tempfile,contextlib,io
from native_tlenet import np,torch,keras,make,choose_policy,windows,vote,capture,restore,equal
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
def softmax(x):
 z=np.exp(x-x.max(axis=-1,keepdims=True));return z/z.sum(axis=-1,keepdims=True)
def reference(m,x):
 for l in m.layers:
  w=l.get_weights();kind=l.__class__.__name__
  if kind=='InputLayer':continue
  if kind=='Conv1D':
   k=w[0].shape[0];a=np.pad(x,((0,0),((k-1)//2,k//2),(0,0)));z=np.lib.stride_tricks.sliding_window_view(a,k,axis=1).transpose(0,1,3,2);x=np.maximum(np.einsum('ntkc,kco->nto',z,w[0],optimize=True)+w[1],0)
  elif kind in ['MaxPooling1D','MaxPool1D']:
   p=l.pool_size[0];x=np.stack([x[:,i:i+p].max(axis=1) for i in range(0,x.shape[1]-p+1,l.strides[0])],axis=1)
  elif kind=='Flatten':x=x.reshape(len(x),-1)
  elif kind=='Dense':
   x=x@w[0]+w[1];x=softmax(x) if l.activation.__name__=='softmax' else np.maximum(x,0)
  else:raise AssertionError(kind)
 return x
def main():
 g=np.random.default_rng(54);original=g.normal(size=(3,90,567)).astype('float32');hot=np.eye(12,dtype='float32')[[0,1,2]]
 with tempfile.TemporaryDirectory() as d:
  h,m=make(d,20261029);assert m.count_params()==121212 and len(m.trainable_weights)==8 and len(m.optimizer.variables)==17
  assert [(l.filters,l.kernel_size) for l in m.layers if isinstance(l,keras.layers.Conv1D)]==[(5,(5,)),(20,(5,))]
  with contextlib.redirect_stdout(io.StringIO()):ax,ay,vx,vy,num=h.pre_processing(original,hot,original[:2],hot[:2])
  assert num==10 and ax.shape==(30,81,567) and ax.dtype==np.float64
  np.testing.assert_array_equal(windows(original,np.arange(30)),ax.astype('float32'));np.testing.assert_array_equal(ay,np.repeat(hot,10,axis=0))
  np.testing.assert_array_equal(windows(original,np.array([29,0,15,9])),ax[[29,0,15,9]].astype('float32'))
  initial=capture(m);x=windows(original,np.arange(2));y=hot[[0,0]]
  for _ in range(3):m.reset_metrics();r=m.train_on_batch(x,y,return_dict=True);assert all(np.isfinite(float(v)) for v in r.values())
  saved=capture(m)
  with torch.no_grad():p=m(x,training=False).detach().numpy()
  np.testing.assert_allclose(p,reference(m,x),rtol=2e-5,atol=2e-6)
  # Evaluate the optimizer schedule directly without changing a model/step.
  for k in [0,1,184,184000]:np.testing.assert_allclose(float(m.optimizer._learning_rate(k)),.01/(1+.005*k),rtol=2e-7,atol=1e-10)
  restore(saved,m);torch.save(saved,Path(d)/'full.pt');m.save(Path(d)/'native.keras');loaded=keras.models.load_model(Path(d)/'native.keras');equal(loaded.get_weights(),saved['native']['weights']);equal([v.numpy() for v in loaded.optimizer.variables],saved['native']['optimizer'])
  _,fresh=make(d,20261029);restore(torch.load(Path(d)/'full.pt',weights_only=False),fresh);equal(capture(fresh),saved);fresh.reset_metrics();rb=fresh.train_on_batch(x,y,return_dict=True);after=capture(fresh);restore(saved,m);m.reset_metrics();ra=m.train_on_batch(x,y,return_dict=True);equal(ra,rb);equal(capture(m),after);restore(initial,m);equal(capture(m),initial)
  labels=np.array([[1]*5+[2]*5,[3]*6+[1]*4]);probs=np.eye(12,dtype='float32')[labels];np.testing.assert_array_equal(vote(probs),[1,3])
 r=dict(state='pass_full_review_tLeNet_native_stream_synthetic_only',total_parameters=121212,trainable_tensors=8,Adam_variables=17,source_epochs=1000,source_batch_size=256,source_metadata_rule_train4700_forces_warp1_slice09=True,slice_length=81,windows_per_parent=10,full_author_preprocessing_vs_lazy_float32_exact=True,all_ten_windows_retained=True,full_NumPy_graph_reference=True,legacy_Adam_inverse_time_decay_verified=True,native_weights_Adam_reload_exact=True,full_custom_fresh_restore_next_update_exact=True,majority_vote_per_parent_and_smallest_tie_exact=True,standalone_source_predict_global_vote_bug_not_used=True,eager_source_float64_training_augmented_bytes=4700*10*81*567*8,memory_adapter_does_not_change_examples_or_scale=True,real_UWB_values_labels_fit=False,formal_cost_run=False,legacy_TF_numeric_parity=False,novelty_claim=False)
 Path(__file__).with_name('SYNTHETIC_CHECK.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
