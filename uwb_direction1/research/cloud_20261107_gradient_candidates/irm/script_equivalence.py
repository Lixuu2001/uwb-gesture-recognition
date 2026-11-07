"""One generated TorchScript equivalence probe, never trace training or load UWB data."""
from pathlib import Path
import json,time,copy,traceback
import native_engine as e
from irm_core import objective
P=Path(__file__).resolve().parent

class ScriptableNativeGAP(e.torch.nn.Module):
    def __init__(self,model):
        super().__init__();self.feature_extractor=model.feature_extractor;self.head=model.head
    def forward(self,x:e.torch.Tensor)->e.torch.Tensor:
        return self.head(self.feature_extractor(x.transpose(1,2)).mean(-1))

def main():
    t=e.torch;start=time.monotonic();cpu=time.process_time()
    try:
        eager,_=e.make();other,_=e.make();scripted=t.jit.script(ScriptableNativeGAP(other))
        compilewall=time.monotonic()-start;compilecpu=time.process_time()-cpu
        assert list(eager.state_dict())==list(scripted.state_dict()) and sum(p.numel() for p in scripted.parameters())==617612
        x=t.randn(64,90,567,generator=t.Generator().manual_seed(7821));y=t.arange(64)%12;d=t.arange(64)//16
        rng=t.get_rng_state().clone();eager.train();a=eager(x);loss,_,_,_=objective(a,y,d,100.);loss.backward();after=t.get_rng_state().clone()
        t.set_rng_state(rng);scripted.train();b=scripted(x);otherloss,_,_,_=objective(b,y,d,100.);otherloss.backward()
        assert t.equal(a,b) and t.equal(after,t.get_rng_state())
        assert all(t.equal(x.grad,z.grad) for x,z in zip(eager.parameters(),scripted.parameters()))
        assert e.semantic(eager.state_dict())==e.semantic(scripted.state_dict())
        eager.eval();scripted.eval()
        with t.no_grad():assert t.equal(eager(x),scripted(x))
        result={'state':'bitwise_generated_native_train_eval_gradients_BN_RNG_script_equivalent','gate_passed':True,'compile_wall_seconds':compilewall,'compile_CPU_seconds':compilecpu,'parameters':617612,'real_UWB_samples_or_labels_read':False,'new_real_fit_started':False}
    except Exception as err:
        (P/'SCRIPT_EQUIVALENCE_STDERR.log').write_text(traceback.format_exc())
        result={'state':'script_equivalence_probe_failed_no_optimized_timing_or_fit','gate_passed':False,'error':repr(err),'wall_seconds':time.monotonic()-start,'CPU_seconds':time.process_time()-cpu,'real_UWB_samples_or_labels_read':False,'new_real_fit_started':False}
    (P/'SCRIPT_EQUIVALENCE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
