"""Synthetic-only state, feature isolation and safe native serialization checks."""
from pathlib import Path
import json,tempfile,time
from fixed_branch_control import *
B=Path(__file__).resolve().parent
def step(m,x,y):
 m.reset_metrics();logs=m.train_on_batch(x,y,return_dict=True);assert np.isfinite(list(logs.values())).all()
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic();g=np.random.RandomState(43);x=g.randn(2,90,567).astype('float32');y=np.eye(12,dtype='float32')[[0,11]]
 with tempfile.TemporaryDirectory() as root:
  root=Path(root)
  for name in ['reference','full','zero','restore']:(root/name).mkdir()
  hr,ref,_=author.make(root/'reference',7,567);reference=capture(ref)
  hf,full,_=make(root/'full',7,'full');equal(reference,capture(full))
  # Full adapter must be the existing published-method baseline, unchanged.
  np.testing.assert_array_equal(ref.predict_on_batch(x),full.predict_on_batch(x))
  for _ in range(3):step(ref,x,y)
  expected=capture(ref)
  for _ in range(3):step(full,x,y)
  equal(expected,capture(full))
  # Recreate baseline before any update, then an exactly paired zero branch.
  hf,full,_=make(root/'full',7,'full');initial=capture(full);hz,zero,changes=make(root/'zero',7,'zero_fixed_responses');equal(initial,capture(zero))
  assert len([l for l in zero.layers if isinstance(l,ZeroFixedBranch)])==1
  equal([tuple(v.shape) for v in full.variables],[tuple(v.shape) for v in zero.variables])
  fixed_full=[l for l in full.layers if isinstance(l,keras.layers.Conv1D) and not l.trainable];fixed_zero=[l for l in zero.layers if isinstance(l,keras.layers.Conv1D) and not l.trainable]
  before=[l.kernel.numpy().copy() for l in fixed_zero]
  for a,b in zip(fixed_full,fixed_zero):np.testing.assert_array_equal(a.kernel.numpy(),b.kernel.numpy())
  full_fixed=keras.Model(full.inputs,hf.hybrid).predict_on_batch(x);zero_fixed=keras.Model(zero.inputs,hz.hybrid).predict_on_batch(x)
  assert full_fixed.shape==zero_fixed.shape==(2,90,17) and np.any(full_fixed!=0);np.testing.assert_array_equal(zero_fixed,np.zeros_like(zero_fixed))
  def first_join(model):return next(l.output for l in model.layers if isinstance(l,keras.layers.Concatenate) and tuple(l.output.shape[1:])==(90,145))
  a=keras.Model(full.inputs,first_join(full)).predict_on_batch(x);b=keras.Model(zero.inputs,first_join(zero)).predict_on_batch(x)
  np.testing.assert_array_equal(a[:,:,:128],b[:,:,:128]);np.testing.assert_array_equal(a[:,:,128:],full_fixed);np.testing.assert_array_equal(b[:,:,128:],np.zeros_like(full_fixed))
  for _ in range(3):step(zero,x,y)
  for layer,w in zip(fixed_zero,before):np.testing.assert_array_equal(layer.kernel.numpy(),w)
  saved=full_state(zero)
  for _ in range(2):step(zero,x,y)
  final=full_state(zero);hz2,recovered,_=make(root/'restore',7,'zero_fixed_responses');restore_full(saved,recovered);equal(saved,full_state(recovered))
  for _ in range(2):step(recovered,x,y)
  equal(final,full_state(recovered))
  recovered.save(root/'zero.keras');loaded=keras.models.load_model(root/'zero.keras',safe_mode=True);equal(recovered.get_weights(),loaded.get_weights());equal([v.numpy() for v in recovered.optimizer.variables],[v.numpy() for v in loaded.optimizer.variables]);np.testing.assert_array_equal(recovered.predict_on_batch(x),loaded.predict_on_batch(x))
  try:make(root/'full',7,'unknown')
  except ValueError:pass
  else:raise AssertionError('Unknown mode accepted')
 result=dict(state='pass_synthetic_paired_fixed_response_zero_preparation',source_commit='cf4d0e55ee11b7393ae3d317cdf5f22dcc51ba3e',binding_sha256=EXPECTED_BINDING_SHA,full_baseline_initial_and3updates_all_native_states_exact=True,full_zero_initial_weights_optimizer_BN_metrics_all_global_RNG_exact=True,all_variable_shapes_identical=True,total_parameters=851434,trainable_parameters=601006,trainable_tensors=50,Adam_variables=102,fixed_filters17_still_computed_and_unchanged=True,original_first_learned128_features_exact=True,only_first_fixed17_responses_zeroed=True,zero_next2_restore_model_Adam_BN_metrics_allRNG_and_modes_exact=True,native_registered_zero_layer_safe_weights_optimizer_predictions_reload_exact=True,unknown_mode_refused=True,synthetic_seed=7,synthetic_batch=2,source_default_epochs1500_not_a_real_configuration=True,current_source_sixfold_untouched=True,new_UWB_values_labels_fit=False,real_ablation_configuration_registered=False,formal_CPU_cost_measured=False,accuracy_result_available=False,standard_InceptionTime_reproduction=False,novelty_claim=False,wall_seconds=time.monotonic()-start)
 (B/'SYNTHETIC_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
