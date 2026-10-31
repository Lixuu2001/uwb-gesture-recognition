"""Synthetic native reload regression; no UWB values/labels/fit."""
import sys,tempfile,json,random,copy
from pathlib import Path
sys.path.insert(0,'/workspace/uwb-hinception-pilot-20261031')
from ops import *
from inspection_rng import isolate_inspection_python_rng
B=Path(__file__).resolve().parent
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 identity={'fixture':'synthetic_native_reload_only'}
 with tempfile.TemporaryDirectory() as d:
  h,m,cb=make(d,7);g=np.random.RandomState(9);x=g.randn(2,90,567).astype('float32');y=np.eye(12,dtype='float32')[[0,1]]
  for _ in range(3):m.train_on_batch(x,y)
  m.save(Path(d)/'fixture.keras');s=snapshot(h,m,0,3,2,identity)
  loaded=keras.models.load_model(Path(d)/'fixture.keras');equal(loaded.get_weights(),s['native']['weights']);equal([v.numpy() for v in loaded.optimizer.variables],s['native']['optimizer']);assert random.getstate()!=s['native']['python_rng']
  restore(s,h,m,expected_identity=identity)
  with isolate_inspection_python_rng(h,m,0,3,2,identity):
   loaded=keras.models.load_model(Path(d)/'fixture.keras');equal(loaded.get_weights(),s['native']['weights']);equal([v.numpy() for v in loaded.optimizer.variables],s['native']['optimizer'])
  equal(s,snapshot(h,m,0,3,2,identity))
  # This island is not permission to overwrite or ignore scientific state.
  try:
   with isolate_inspection_python_rng(h,m,0,3,2,identity):m.optimizer.learning_rate.assign(.123)
  except AssertionError:pass
  else:raise AssertionError('Unexpected model/optimizer mutation hidden')
  restore(s,h,m,expected_identity=identity);equal(s,snapshot(h,m,0,3,2,identity));close(cb)
 result=dict(state='pass_synthetic_native_reload_observational_regression',unisolated_native_reload_consumes_python_rng=True,isolated_native_reload_all_model_Adam_BN_metrics_callbacks_modes_and_all_RNG_exact=True,unexpected_optimizer_mutation_still_rejected=True,only_inspection_python_rng_restored=True,comparison_tolerances_relaxed=False,real_UWB_values_labels_or_fit=False)
 (B/'REGRESSION_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
