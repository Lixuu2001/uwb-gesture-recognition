"""Deny real arrays/IDs during imports and a complete native generated update; no cost reprofile."""
from pathlib import Path
import sys,json,time
P=Path(__file__).resolve().parent;blocked=[]
def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        p=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).absolute();s=str(p)
        if s.startswith('/workspace/uwb-restored-') or (s.startswith('/workspace/') and p.suffix in ['.npy','.npz'] and not p.is_relative_to(P)):
            blocked.append(s);raise AssertionError('Real UWB arrays/IDs forbidden in generated core audit')
sys.addaudithook(audit)
import torch
import torch.nn.functional as F
import numpy as np
from groupdro_core import GroupDROState,NativeGroupDROOptimizer
sys.path.insert(0,'/workspace/uwb-timemil-20261029');import adapter as native
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
started=time.monotonic();cpu=time.process_time();torch.manual_seed(19)
rows=[]
for h in range(1,7):
    val=h%6+1;train=tuple(u for u in range(1,7) if u not in [h,val]);q=GroupDROState(train)
    z=torch.randn(64,12,requires_grad=True);y=torch.arange(64)%12;d=torch.as_tensor(train)[torch.arange(64)%4];loss,_,info=q.objective(z,y,d);loss.backward();assert set(info['domain_users'])==set(train) and not {h,val}&set(info['domain_users'])
    before=q.state_dict();bad=d.clone();bad[0]=h
    try:q.objective(z,y,bad)
    except AssertionError:pass
    else:raise AssertionError('Held user entered q')
    assert q.updates==before['updates'] and torch.equal(q.q,before['q'])
    rows.append(dict(held=h,validation=val,training_groups=list(train),held_rejection_without_q_mutation=True))
q=GroupDROState((3,4,5,6));z=torch.randn(1,12,requires_grad=True);y=torch.tensor([2]);loss,base,_=q.objective(z,y,torch.tensor([4]));torch.testing.assert_close(loss,base);grad=torch.autograd.grad(loss,z)[0];expected=(torch.sigmoid(z)-F.one_hot(y,12))/12;torch.testing.assert_close(grad,expected)
before=q.state_dict()
try:q.objective(torch.full((1,12),float('nan')),y,torch.tensor([4]))
except AssertionError:pass
else:raise AssertionError('Nonfinite risk accepted')
assert q.updates==before['updates'] and torch.equal(q.q,before['q'])
m=native.make_model('gap',20261029);assert sum(p.numel() for p in m.parameters())==617612
o=NativeGroupDROOptimizer(m.parameters(),(3,4,5,6),lambda ps:native.StableLookahead(ps,lr=.001,weight_decay=.0001));x=torch.randn(64,90,567);y=torch.arange(64)%12;d=torch.tensor([3,4,5,6]*16);z=m(x);loss,_,_=o.robust.objective(z,y,d);o.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2.);o.step();assert o.native.count==o.robust.updates==1 and not blocked
result=dict(state='real_array_and_ID_open_firewall_native_generated_full_update_and_all6_group_roles_passed',actual_UWB_array_or_ID_open_events=0,real_UWB_fit_started=False,parameters=617612,rows=rows,single_present_group_equals_globalBCE_value_and_gradient=True,nonfinite_risk_rejected_without_q_mutation=True,cost_profile_repeated=False,wall=time.monotonic()-started,CPU=time.process_time()-cpu)
(P/'GENERATED_FIREWALL_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
