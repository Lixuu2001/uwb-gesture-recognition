"""Source validation-loss checkpoint rule, synthetic state changes only."""
import tempfile,json,hashlib
from native_disjoint import *
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 with tempfile.TemporaryDirectory() as d:
  h,m=make(d,7);assert configure_from_source(h,25,4,8)==2;cb=keras.callbacks.CallbackList(h.callbacks,add_history=False,add_progbar=False,model=m,epochs=4,steps=1,verbose=0);cb.on_train_begin();target=Path(d)/'best_model.keras';saved=None
  for ep,(train,val) in enumerate([(4.,8.),(.5,7.),(.1,7.),(.01,7.5)]):
   m.layers[-1].bias.assign(np.full(12,ep,dtype='float32'));cb.on_epoch_end(ep,{'loss':train,'val_loss':val});digest=hashlib.sha256(target.read_bytes()).hexdigest()
   if ep==1:saved=digest
   if ep>1:assert digest==saved
  loaded=keras.models.load_model(target);np.testing.assert_array_equal(loaded.layers[-1].bias.numpy(),np.ones(12,dtype='float32'));cb.on_train_end()
  if cb._executor is not None:cb._executor.shutdown(wait=True);cb._executor=None
 result=dict(state='pass_synthetic_callback_rule_only',source_original_fit_prefix=True,selection_strict_minimum_val_loss=True,tied_validation_retains_first=True,lower_training_loss_does_not_override_worse_or_tied_validation=True,no_optimizer_updates=True,real_UWB_values_labels_or_fit=False);(B/'CHECKPOINT_RULE_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
