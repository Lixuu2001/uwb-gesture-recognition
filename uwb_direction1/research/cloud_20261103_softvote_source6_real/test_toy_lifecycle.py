"""Full64/100 control paths with a tiny explicit synthetic backend.

Actual default-model numeric/resume tests are test_native_state.py. These
fixtures verify selection, journals, first-step release, crash/suffix recovery
and independent replay without spending another full real training trajectory.
"""
import unittest,tempfile,sys,random,copy
from pathlib import Path
from unittest.mock import patch
import contextlib
import engine as e
e.setup('shuffle');e.torch.set_num_threads(1)
import common,produce,first_audit,audit_inference,replay_shuffle,checks,controller,journal
sys.path.insert(0,'/workspace/uwb-timemil-20261029')
from adapter import StableLookahead
class Toy(e.torch.nn.Module):
    def __init__(self):
        super().__init__();self.fc=e.torch.nn.Linear(3,12);self.register_buffer('updates_count',e.torch.tensor(0))
    def forward(self,x):return self.fc(x.reshape(len(x),-1)[:,:3])
def make(arm):
    random.seed(20261029);e.np.random.seed(20261029);e.torch.manual_seed(20261029)
    m=Toy();o=StableLookahead(m.parameters(),lr=.001,weight_decay=.0001) if arm=='timemil' else e.torch.optim.Adam(m.parameters(),lr=.001,foreach=False,fused=False)
    return m,o,e.torch.Generator().manual_seed(20261029)
def step(arm,m,o,x,y,epoch):
    o.zero_grad();z=m(e.torch.from_numpy(x));target=e.torch.as_tensor(y).long()
    loss=e.torch.nn.functional.binary_cross_entropy_with_logits(z,e.torch.nn.functional.one_hot(target,12).float()) if arm=='timemil' else e.torch.nn.functional.cross_entropy(z,target)
    loss.backward()
    if arm=='timemil':e.torch.nn.utils.clip_grad_norm_(m.parameters(),2.)
    o.step();m.updates_count+=1;return float(loss.detach())
def evaluate(arm,m,maps,positions,y=None):
    m.eval();count=int(m.updates_count);scores=e.np.zeros((len(positions),12),dtype='float32');scores[:,0]=count%7
    if y is None:return scores
    from data import macro
    return scores,macro(y,scores.argmax(1)),(count%7)+.1
class Lifecycle(unittest.TestCase):
    def run_case(self,arm,crash):
        with tempfile.TemporaryDirectory() as folder:
            b=Path(folder);ids=e.np.arange(80);users=e.np.ones(80,dtype='int64');loc=dict(train=e.np.arange(64),validation=e.np.arange(64,72),diagnostic=e.np.arange(72,80));maps=e.np.ones((80,90,567),dtype='float32')
            y=e.np.arange(64)%12;v=e.np.arange(8)%12
            c=dict(first_update_release_timeout_seconds=10)
            fake_load=lambda held,labels=False:(ids,users,loc,maps,y if labels else None,v if labels else None)
            with contextlib.ExitStack() as stack:
                for mod in [common,produce,first_audit,audit_inference,checks,controller]:stack.enter_context(patch.object(mod,'B',b))
                for mod in [produce,first_audit,audit_inference,replay_shuffle]:
                    stack.enter_context(patch.object(mod,'require_initial',lambda:(c,{})));stack.enter_context(patch.object(mod,'load',fake_load))
                stack.enter_context(patch.object(e,'make',make));stack.enter_context(patch.object(e,'step',step));stack.enter_context(patch.object(e,'evaluate',evaluate))
                m,o,g=make(arm);s=e.capture(m,o,g,0,0);common.write(b/'SEED_FILES.json',{arm:dict(state_sha256=e.semantic(s))})
                def release(_):
                    out=checks.member(2,arm)
                    if (out/'FIRST_READY.json').exists() and not (out/'FIRST_RELEASE.json').exists():
                        with patch.object(sys,'argv',['first_audit.py','2',arm]):first_audit.main()
                        common.write(out/'first_audit/TERMINATION.json',dict(exit_code=0,resource_stop_reason=None,cgroup_OOM_kill_delta=0,synthetic_fixture=True));controller.release_first(out)
                stack.enter_context(patch.object(produce.time,'sleep',release))
                original=journal.StepLog.append
                def interrupted(self,row):
                    original(self,row)
                    if row['update']==9:raise RuntimeError('Synthetic interruption after persisted uncommitted row9')
                args=['produce.py','2',arm]
                if crash:
                    with patch.object(journal.StepLog,'append',interrupted),patch.object(sys,'argv',args):
                        with self.assertRaises(RuntimeError):produce.main()
                    prefix=(checks.member(2,arm)/'steps.jsonl').read_bytes()
                    with patch.object(sys,'argv',args+['--resume']):produce.main()
                    self.assertTrue((checks.member(2,arm)/'steps.jsonl').read_bytes().startswith(prefix))
                    self.assertTrue((checks.member(2,arm)/'SUFFIX_REPLAY_VERIFIED.json').exists())
                else:
                    with patch.object(sys,'argv',args):produce.main()
                out=checks.member(2,arm)
                with patch.object(sys,'argv',['audit_inference.py','2',arm]):audit_inference.main()
                self.assertEqual(common.read(out/'inference_audit/CERTIFICATE.json')['state'],'pass')
                if arm=='timemil':self.assertEqual(common.read(out/'ARM_COMPLETE.json')['selected_epoch'],7)
                else:
                    original_commit=replay_shuffle.commit
                    def pause(*args,**kwargs):
                        original_commit(*args,**kwargs)
                        if args[1]['epoch']==3:raise RuntimeError('Synthetic own-audit checkpoint interruption')
                    with patch.object(replay_shuffle,'commit',pause),patch.object(sys,'argv',['replay_shuffle.py','2']):
                        with self.assertRaises(RuntimeError):replay_shuffle.main()
                    with patch.object(sys,'argv',['replay_shuffle.py','2']):replay_shuffle.main()
                    cert=common.read(out/'replay/COMPLETE_AUDIT.json');self.assertTrue(cert['all100epochs_all_updates_replayed_exactly'])
    def test_TimeMIL64_selection_and_committed_recovery(self):self.run_case('timemil',True)
    def test_Shuffle100_producer_suffix_and_own_audit_recovery(self):self.run_case('shuffle',True)
if __name__=='__main__':unittest.main(verbosity=2)
