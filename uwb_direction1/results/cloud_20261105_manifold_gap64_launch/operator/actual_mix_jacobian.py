"""Independent closed-form gradient check on the ACTUAL mixing function."""
from pathlib import Path
import sys,json,hashlib
S=Path('/workspace/uwb-manifold-gap64-20261105');sys.path.insert(0,str(S))
import engine as e
from common import write
cases=[]
for seed in range(10):
    rng=e.np.random.default_rng(seed);x=e.torch.tensor(rng.normal(size=(7,5)),dtype=e.torch.float64,requires_grad=True);w=e.torch.tensor(rng.normal(size=(7,5)),dtype=e.torch.float64)
    target=e.torch.nn.functional.one_hot(e.torch.arange(7)%12,12).double();lam=e.torch.tensor([.37],dtype=e.torch.float64)
    e.np.random.seed(seed);mixed,soft,p=e.mix_at(x,target,lam)
    actual=e.torch.autograd.grad((mixed*w).sum(),x)[0]
    expected=lam*w+(1-lam)*w[e.np.argsort(p)]
    e.torch.testing.assert_close(actual,expected,rtol=1e-14,atol=1e-14)
    cases.append(dict(seed=seed,non_involutive=not e.np.array_equal(p[p],e.np.arange(7)),maximum_gradient_error=float((actual-expected).abs().max())))
assert any(r['non_involutive'] for r in cases)
q=dict(state='pass_actual_mix_at_general_permutation_closed_form_Jacobian',cases=cases,actual_engine_sha256=hashlib.sha256((S/'engine.py').read_bytes()).hexdigest(),no_UWB_arrays_or_labels_opened=True,model_or_scientific_source_modified=False,actual_training_process_rng_untouched=True)
write(S/'ADDITIONAL_ACTUAL_MIX_JACOBIAN.json',q);print(json.dumps(q))
