"""Original held1 initialization and variable-count synthetic state recovery."""
import tempfile,json,sys
from pathlib import Path
from ops import *
import ops
from inspection_rng import isolate_inspection_python_rng
B=Path(__file__).resolve().parent
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 with tempfile.TemporaryDirectory() as d:
  ops.configure(1,4700);h,m,cb=make(d,20261029);old=torch.load('/workspace/uwb-hinception-pilot-20261031/run01/initial.pt',map_location='cpu',weights_only=False);initial=snapshot(h,m,0,0,-1,old['identity']);equal(old['native'],initial['native']);equal(old['callbacks'],initial['callbacks']);equal(old['module_training_modes'],initial['module_training_modes']);close(cb)
 with tempfile.TemporaryDirectory() as d,tempfile.TemporaryDirectory() as e:
  ops.configure(3,4500);h,m,cb=make(d,7);g=np.random.RandomState(11);x=g.randn(2,90,567).astype('float32');y=np.eye(12,dtype='float32')[[0,1]];identity={'fixture':'synthetic_variable_count_state_only'}
  for _ in range(3):m.train_on_batch(x,y)
  saved=snapshot(h,m,1,3,2,identity)
  for _ in range(2):m.train_on_batch(x,y)
  expected=snapshot(h,m,1,5,4,identity);close(cb)
  h2,m2,cb2=make(e,7);restore(saved,h2,m2,expected_identity=identity);equal(saved,snapshot(h2,m2,1,3,2,identity))
  for _ in range(2):m2.train_on_batch(x,y)
  equal(expected,snapshot(h2,m2,1,5,4,identity));m2.save(Path(e)/'native.keras')
  with isolate_inspection_python_rng(h2,m2,1,5,4,identity):
   loaded=keras.models.load_model(Path(e)/'native.keras');equal(loaded.get_weights(),m2.get_weights());equal([v.numpy() for v in loaded.optimizer.variables],[v.numpy() for v in m2.optimizer.variables])
  equal(expected,snapshot(h2,m2,1,5,4,identity));ops.configure(4,4500)
  try:restore(saved,h2,m2,expected_identity=identity)
  except ValueError:pass
  else:raise AssertionError('Cross-fold checkpoint accepted')
  close(cb2)
 # Metadata-only sampler coverage for all variable counts; no real row IDs.
 for held,count in [(2,4700),(3,4500),(4,4500),(5,4600),(6,4600)]:
  ops.configure(held,count);order=training_order(1);assert len(order)==count and len(np.unique(order))==count and order.min()==0 and order.max()==count-1
 for count in [1000,1100,1200]:
  for role in ['validation','diagnostic']:
   order=evaluation_order(count,role);assert len(np.unique(order))==count and np.array_equal(np.sort(order),np.arange(count));equal(order,evaluation_order(count,role))
 result=dict(state='pass_original_initialization_and_synthetic_variable_count_restore',original_held1_model_Adam_BN_metrics_RNG_callbacks_modes_initialization_exact=True,full_model_851434_601006_unchanged=True,restore_after3_next2_all_states_exact=True,observational_native_reload_isolated_and_all_states_exact=True,cross_fold_state_refused=True,all_variable_count_sampler_coverage=True,new_UWB_values_labels_fit=False)
 (B/'NATIVE_STATE_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
