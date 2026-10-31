"""Read-only final-state diagnosis; no UWB arrays/labels or optimizer updates."""
import sys,json,tempfile,hashlib,copy
from pathlib import Path
STUDY=Path('/workspace/uwb-hinception-pilot-20261031');sys.path.insert(0,str(STUDY))
from ops import *
B=Path(__file__).resolve().parent
def same(a,b):
 try:equal(a,b);return True
 except AssertionError:return False
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 out=STUDY/'run01';last=torch.load(out/'last.pt',map_location='cpu',weights_only=False);final=torch.load(out/'final_prediction_state.pt',map_location='cpu',weights_only=False);best=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
 result=dict(saved_last_to_final={k:same(last['native'][k],final['native'][k]) for k in last['native'] if k!='weights'},saved_other_fields={k:same(last[k],final[k]) for k in last if k!='native'})
 with tempfile.TemporaryDirectory() as d:
  h,m,cb=make(d);restore(last,h,m,expected_identity=last['identity']);equal(last,snapshot(h,m,100,7400,73,last['identity']))
  m.save(Path(d)/'mirrored_producer_last.keras');after_save=snapshot(h,m,100,7400,73,last['identity'])
  result['native_save_last_changed_fields']=[k for k in last['native'] if not same(last['native'][k],after_save['native'][k])]
  # Replay original auditor's inspection; this must be observational.
  inspect_before=snapshot(h,m,100,7400,73,last['identity'])
  for name,s in [('best_model.keras',best),('last_model.keras',last)]:
   loaded=keras.models.load_model(out/name);equal(loaded.get_weights(),s['native']['weights']);equal([v.numpy() for v in loaded.optimizer.variables],s['native']['optimizer'])
  inspect_after=snapshot(h,m,100,7400,73,last['identity']);result['native_reload_changed_fields']=[k for k in inspect_before['native'] if not same(inspect_before['native'][k],inspect_after['native'][k])]
  result['native_reload_changed_non_native_fields']=[k for k in inspect_before if k!='native' and not same(inspect_before[k],inspect_after[k])]
  result['native_reload_changed_model_optimizer_metrics']=[k for k in ['weights','optimizer','metrics'] if not same(inspect_before['native'][k],inspect_after['native'][k])]
  # Isolate only inspection RNG, then use selected weights exactly as producer.
  import random
  from keras.src.random.seed_generator import global_seed_generator
  rng=inspect_before['native'];random.setstate(rng['python_rng']);np.random.set_state(rng['numpy_rng']);torch.set_rng_state(rng['torch_rng']);global_seed_generator().state.assign(rng['keras_global_rng'])
  equal(inspect_before,snapshot(h,m,100,7400,73,last['identity']));m.set_weights(best['native']['weights']);before_prediction=snapshot(h,m,100,7400,73,last['identity'])
  result['restored_inspection_rng_then_selected_weights_equals_saved_final_without_real_forward']=same(final,before_prediction)
  # H-Inception has no inference stochastic layers; prove any remaining field.
  result['remaining_fields_before_real_prediction']=[k for k in final['native'] if not same(final['native'][k],before_prediction['native'][k])]
  close(cb)
 result.update(state='read_only_checkpoint_diagnosis',real_UWB_arrays_labels_or_forward_fit=False,diagnostic_summary_parsed=False)
 (B/'RNG_DIAGNOSIS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
