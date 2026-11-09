"""Generated native lookahead-cycle and old frozen unscored output compatibility."""
from pathlib import Path
import copy,hashlib
import core as e
from lazy_outer_data import OuterInputs
from worker import evaluate
np=e.np;torch=e.torch
def main():
    m,o,t=e.make();plain,po,_=e.make()
    rng=np.random.default_rng(81);x=rng.normal(size=(2,90,567)).astype('float32');y=np.array([0,11],dtype=np.int64)
    cycles=[]
    for i in range(12):
        before=torch.get_rng_state().clone();row=e.step(m,o,t,x,y,True);after=torch.get_rng_state().clone()
        torch.set_rng_state(before);plain.train();z=plain(torch.from_numpy(x));po.zero_grad();loss=e.bce(z,y);loss.backward();torch.nn.utils.clip_grad_norm_(plain.parameters(),2.);po.step()
        assert torch.equal(after,torch.get_rng_state()) and e.semantic(m.state_dict())==e.semantic(plain.state_dict()) and e.semantic(o.state_dict())==e.semantic(po.state_dict())
        if (i+1)%6==0:
            assert len(o.slow)==len(list(m.parameters()));cycles.append(i+1)
    # Resume through a lookahead boundary, including model/dropout RNG, optimizer and collector state.
    state=e.capture(m,o,t,0);before=torch.get_rng_state().clone();a=e.step(m,o,t,x,y,True)
    fresh,fo,ft=e.make();e.restore(state,fresh,fo,ft);b=e.step(fresh,fo,ft,x,y,True);assert a==b
    # This tests the new lazy/role evaluation interface against the already frozen original GAP held1 output.
    # No gesture labels are loaded, and no method/tail held-user prediction or quality score is computed.
    root=Path('/workspace/uwb-restored-20261103');d=OuterInputs(root/'cache/X_clutterremoved.npy',root/'cache/subj_clutterremoved.npy','/workspace/uwb-accuracy-loop-20261029/run01/source_ids.npz')
    row=e.read('/workspace/uwb-swad-loso6-20261108/folds/held1/PROTOCOL.json')['baseline_artifacts']
    path=Path(row['model']);assert e.sha(path)==row['model_sha256']
    assert set(torch.serialization.get_unsafe_globals_in_checkpoint(path))<={'torch.torch_version.TorchVersion'}
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):saved=torch.load(path,map_location='cpu',weights_only=True)
    model,_,_=e.make();model.load_state_dict(saved['model']);model.eval()
    class View:
        roles=d.roles(1)
        def batch(self,role,ids):return d.batch(1,role,ids)
    output,macro,loss=evaluate(model,View(),'test');assert macro is None and loss is None
    pred=Path(row['prediction']);assert e.sha(pred)==row['prediction_sha256']
    with np.load(pred,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],View.roles['sample_ids']['test']);np.testing.assert_array_equal(z['logits'],output)
    e.write(e.ROOT/'NATIVE_OUTPUT_AND_OPTIMIZER_AUDIT.json',dict(passed=True,generated_native_passive_and_plain_model_optimizer_rng_equal_each_update=True,generated_lookahead_cycle_boundaries=cycles,generated_resume_state_exact=True,old_original_GAP_held1_model_and1200_unscored_logits_lazy_worker_interface_bitwise_equal=True,old_model_sha256=row['model_sha256'],old_prediction_sha256=row['prediction_sha256'],gesture_truth_values_read=False,new_method_held_predictions_or_scores=False,users7_8_gesture_truth_read=False,scope='adapter and new worker interface compatibility; no new cohort quality evidence, not independent optimizer derivation; fresh replay cannot exclude every shared implementation error',audit_source_sha256=e.sha(__file__),frozen_source_sha256=e.sha(e.ROOT/'FROZEN_HASHES.json')))
    print('PASS: generated Lookahead6/12 cycles and resume; original native held1 lazy1200 unscored logits exact, no truth')
if __name__=='__main__':main()
